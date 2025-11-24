import os
import time
import shutil
import argparse
import pickle
import torch
import numpy as np
from tqdm import tqdm
from models import arch2relu
from evolution import Mutation, Crossover, init, get_tradeoffs, select
from models import create_model
from datasets import create_dataloader, create_subloader, DATASETS
from lightning.pytorch import seed_everything
import multiprocessing as mp
from parallel import CPUParalCompute
from ckks import coeffs2ckks
from helper import train, validate


parser = argparse.ArgumentParser('AutoFHE: Automated Adaption of CNNs for Efficient Evaluation over FHE')
parser.add_argument('-a', '--arch', metavar='ARCH', type=str, default=None, required=True,
                    help='models: VGG11, VGG16, ResNet20, ResNet32, ResNet44, ResNet56, ResNet110 (default: None)')
parser.add_argument('-d', '--dataset', metavar='DATASET', type=str, default='cifar10',
                    help="datasets: cifar10, cifar100, TinyImageNet (default: cifar10)")
parser.add_argument('--data', metavar='DIR', type=str, default=None, help='path to dataset (default: None)')
parser.add_argument('--ckpt', type=str, metavar='PATH', default=None, help='path to ckpt (default: None)')
parser.add_argument('--seed', metavar='N', type=int, default=0, help='seed for initializing training (default: 0)')
parser.add_argument('--gen', metavar='N', default=10, type=int, help='number of evolution generations (default: 10).')
parser.add_argument('--pop', metavar='N', default=20, type=int, help='population size (default: 20).')
parser.add_argument('--com-no', metavar='N', default=6, type=int, help='number of sub-functions of composite polynomials (default: 6).')
parser.add_argument('--deg-max', metavar='N', default=7, type=int, help='maximum degree of sub-polynomials (default: 7).')
parser.add_argument('--cpus', metavar='N', default=100, type=int, help='CPUs used to solve coefficients (default: 100).')
parser.add_argument('--gpu', type=int, default=None, help='GPU used to Evaluate or Finetune.')
parser.add_argument('--minival', metavar='N', type=int, default=10000, help='num of training images used as minival (default: 10000).')
parser.add_argument('--grad-clip', metavar='F', type=float, default=1., help='gradient clip value (default: 1.)')
parser.add_argument('--alpha', metavar='F', type=float, default=0.9, help='KD alpha (default: 0.9)')
parser.add_argument('--batch-size', metavar='N', default=128, type=int, help='batch size (default: 128)')
parser.add_argument('--epochs', default=5, type=int, metavar='N', help='number of total epochs to finetune (default: 5)')
parser.add_argument('-j', '--num-workers', default=16, type=int, metavar='N', help='number of workers (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.02, type=float, metavar='LR', help='learning rate (default: 0.02)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('-o', '--output', metavar='PATH', default="experiments-search", type=str, help='path to output (default: experiments-search/{arch}-{dataset})')
parser.add_argument('--use-wandb', action='store_true', default=False)
parser.add_argument('--wandb-proj', type=str, metavar='PROJ', default='AutoFHE-Search', help='wandb project (default: AutoFHE-Search)')
parser.add_argument('--resume', action='store_true', default=False)


def main():

    args = parser.parse_args()
    args.arch = args.arch.lower()
    args.dataset = args.dataset.lower()
    args.off = args.pop * 6
    assert os.path.isfile(args.ckpt), f"=> Not found ckpt: {args.ckpt}"
    assert args.pop > (args.deg_max + 1) // 2, "=> Population size is too small"
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    dir_name = f"{args.arch}-{args.dataset}"
    args.output = os.path.join(args.output, dir_name)
    if not args.resume:
        if not os.path.isdir(args.output):
            os.mkdir(args.output)
        else:
            shutil.rmtree(args.output)
            os.mkdir(args.output)
    else:
        assert os.path.isfile(args.output + "/search_ckpt_last"), f'Not found: {args.output + "/search_ckpt_last"}'
    if not os.path.isdir(args.output + "/population"):
        os.mkdir(args.output + "/population")
    if not os.path.isdir(args.output + "/offspring"):
        os.mkdir(args.output + "/offspring")
    if args.seed is not None:
        seed_everything(args.seed)
    if args.cpus is None:
        args.cpus = mp.cpu_count()
    else:
        args.cpus = min(args.cpus, mp.cpu_count())
    print(f'=> {args.cpus} CPUs are used to solve coefficients.')
    device = torch.device("cpu")
    if torch.cuda.is_available() and args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
        print(f'=> GPU {args.gpu} is used to evaluate or finetune.')
    relu_no = arch2relu[args.arch]
    _, num_classes = DATASETS[args.dataset]

    # Model and Dataloader
    assert os.path.isfile(args.ckpt), f"=> Not found: {args.ckpt}"
    model = create_model(args.arch, args.dataset)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict({k.replace("model.", ""): v for k, v in ckpt["state_dict"].items() if "model" in k})
    model.eval()
    model.to(device)
    train_loader, val_loader, minival_loader = create_dataloader(args.dataset, args.data, args.batch_size, args.num_workers, args.minival)
    train_loader_idx, _ = create_dataloader(args.dataset, args.data, args.batch_size, args.num_workers, index=True)
    model.enable_track_relu()
    with torch.no_grad():
        for images, _ in train_loader:
            images = images.to(device)
            model(images)
    model.disable_track_relu()

    # record
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_proj, name=f"{args.arch}-{args.dataset}", config=vars(args))
        all_pop_boot = []
        all_pop_accuracy = []
        all_keys = []

    # Initialize
    end = time.time()
    start_gen = 0
    deg2poly = {}
    mutate = Mutation(relu_no, 0, args.deg_max)
    crossover = Crossover(relu_no)
    if not args.resume:
        population = init(args.pop * 10, relu_no, args.com_no, 0, args.deg_max)
        population[0, :] = 0
        for i in range(1, args.deg_max + 1, 2):
            population[(i+1)//2, :] = i
        population = np.asarray(population, dtype=int)
        population, pop_coeffs, pop_accuracy, pop_boot, pop_depth, pop_ckpts = finetune_population(population, model,
            train_loader, train_loader_idx, minival_loader, deg2poly, num_classes, relu_no, args, args.output+"/population", device=device, prefix="=> Initialize")
        pop_fitness = np.column_stack((pop_boot, 1 - pop_accuracy))
        fronts = get_tradeoffs(pop_fitness, args.pop)
        fronts = np.concatenate(fronts)
        population = population[fronts]
        pop_coeffs = pop_coeffs[fronts]
        pop_accuracy = pop_accuracy[fronts]
        pop_boot = pop_boot[fronts]
        pop_depth = pop_depth[fronts]
        pop_ckpts = pop_ckpts[fronts]

    else:
        population, pop_coeffs, pop_accuracy, pop_boot, pop_depth, pop_ckpts, deg2poly, itr, search_time = pickle.load(open(args.output + "/search_ckpt_last", "rb"))
        start_gen = start_gen + itr + 1
        end = end - search_time
        print(f"=> Resume from generation {itr}")

    for itr in range(start_gen, args.gen):

        for explore, prefix in zip([crossover, mutate], ["crossover", "mutate"]):

            # select
            pop_fitness = np.column_stack((pop_boot, 1 - pop_accuracy))
            fronts = get_tradeoffs(pop_fitness)
            selected = select(fronts, args.off)

            # crossover or mutate
            offspring = population[selected]
            ckpts = pop_ckpts[selected]
            offspring = explore(offspring)

            # fine-tune and evalaute
            if prefix == "crossover":
                ckpts = None
            offspring, off_coeffs, off_accuracy, off_boot, off_depth, off_ckpts = finetune_population(offspring, model,
                       train_loader, train_loader_idx, minival_loader, deg2poly, num_classes, relu_no, args, args.output + "/offspring",
                       device=device, ckpts=ckpts, prefix="=> Generation {:d}| {:}".format(itr, prefix))

            # update population
            population = np.row_stack((population, offspring))
            pop_coeffs = np.row_stack((pop_coeffs, off_coeffs))
            pop_accuracy = np.concatenate((pop_accuracy, off_accuracy))
            pop_boot = np.concatenate((pop_boot, off_boot))
            pop_depth = np.concatenate((pop_depth, off_depth))
            pop_ckpts = np.concatenate((pop_ckpts, off_ckpts))
            pop_fitness = np.column_stack((pop_boot, 1 - pop_accuracy))
            fronts = get_tradeoffs(pop_fitness, args.pop)
            fronts = np.concatenate(fronts)
            population = population[fronts]
            pop_coeffs = pop_coeffs[fronts]
            pop_accuracy = pop_accuracy[fronts]
            pop_boot = pop_boot[fronts]
            pop_depth = pop_depth[fronts]
            pop_ckpts_old = pop_ckpts[fronts]
            pop_ckpts = []
            os.mkdir(args.output + "/temp")
            for i, ckpt_path in enumerate(pop_ckpts_old):
                shutil.copyfile(ckpt_path, args.output + f'/temp/{i}.ckpt')
                pop_ckpts.append(args.output + f'/population/{i}.ckpt')
            pop_ckpts = np.asarray(pop_ckpts, dtype=object)
            shutil.rmtree(args.output + "/population")
            shutil.rmtree(args.output + "/offspring")
            os.rename(args.output + "/temp", args.output + "/population")
            os.mkdir(args.output + "/offspring")

        # checkpoint
        pop_fitness = np.column_stack((pop_boot, 1 - pop_accuracy))
        fronts = get_tradeoffs(pop_fitness)
        pareto = fronts[0]
        pareto_ckpts = pop_ckpts[pareto]
        pareto_accuracy = pop_accuracy[pareto]
        pareto_boot = pop_boot[pareto]
        if os.path.isdir(args.output + "/pareto"):
            shutil.rmtree(args.output + "/pareto")
        os.mkdir(args.output + "/pareto")
        for i, ckpt_path in enumerate(pareto_ckpts):
            shutil.copyfile(ckpt_path, args.output + f'/pareto/{i}.ckpt')
        idx = np.argsort(pareto_boot)
        pareto_accuracy, pareto_boot = pareto_accuracy[idx], pareto_boot[idx]
        pickle.dump((population, pop_coeffs, pop_accuracy, pop_boot, pop_depth, pop_ckpts, deg2poly, itr, time.time()-end), open(args.output + f'/search_ckpt_gen{itr}', "wb"))
        shutil.copyfile(args.output + f'/search_ckpt_gen{itr}', args.output + '/search_ckpt_last')
        log = open(args.output + "/search_log.csv", "a")
        log.write("generation, {:d}, time, {:.2f}\n".format(itr, time.time()-end))
        for b_ in pareto_boot:
            log.write("{:d},".format(b_))
        log.write("\n")
        for a_ in pareto_accuracy:
            log.write("{:.6f},".format(a_))
        log.write("\n")
        log.close()
        if args.use_wandb:
            table = wandb.Table(data=np.column_stack([pareto_boot, pareto_accuracy]), columns=["bootstrapping", "accuracy"])
            line_plot = wandb.plot.line(table, x='bootstrapping', y='accuracy', title=f'Tradeoff Gen {itr}')
            wandb.log({f'tradeoff-gen-{itr}': line_plot})
            all_pop_boot.append(pareto_boot)
            all_pop_accuracy.append(pareto_accuracy)
            all_keys.append(f'Tradeoff Gen {itr}')
            wandb.log({"all_tradeoffs": wandb.plot.line_series(xs=all_pop_boot, ys=all_pop_accuracy,
                keys=all_keys, title="all tradeoffs", xname="bootstrapping")})

    if args.use_wandb:
        wandb.finish()


def finetune_population(population: np.ndarray, model, train_loader, train_loader_idx, minival_loader, deg2poly: dict, num_classes: int,
                        relu_no: int, args, save: str, device=torch.device("cpu"), ckpts: np.ndarray = None, prefix=""):
    if ckpts is not None:
        assert len(population) == len(ckpts), "=> Population size and number of ckpts are not matched"
        if type(ckpts) == list:
            ckpts = np.asarray(ckpts)
        for ckpt_path in ckpts:
            assert os.path.isfile(ckpt_path), f"=> Not found: {ckpt_path}"
    # use RCCDE to optimize coefficients
    pop_degrees = []
    for x in population:
        pop_degrees.extend(np.array_split(x, relu_no))
    unq_degrees = {}
    for deg in pop_degrees:
        k = np.array2string(deg)
        if k not in deg2poly.keys() and k not in unq_degrees.keys():
            unq_degrees[k] = deg
    cpucompute = CPUParalCompute(args.cpus, unq_degrees, prefix=prefix)
    cpucompute()
    unq_polys = cpucompute.get()
    deg2poly.update(unq_polys)
    NP = len(population)
    pop_polys = np.empty((NP, relu_no), dtype=list)
    for i in range(NP):
        degrees = np.array_split(population[i], relu_no)
        for j in range(relu_no):
            deg_key = np.array2string(degrees[j])
            if deg_key in deg2poly.keys():
                pop_polys[i][j] = deg2poly[deg_key]
    valid = np.asarray([all(pop_polys[i]) for i in range(NP)], dtype=bool)
    population = population[valid]
    if ckpts is not None:
        ckpts = ckpts[valid]

    # bootstrapping consumption
    pop_coeffs = []
    pop_boot = []
    pop_depth = []
    valid = []
    for polys in pop_polys:
        coeffs = [[f.coef for f in p] for p in polys]
        ckks = coeffs2ckks(args.arch, args.dataset, coeffs)
        if ckks:
            boot, dep = ckks[0], ckks[1]
            pop_coeffs.append(coeffs)
            pop_boot.append(boot)
            pop_depth.append(dep)
            valid.append(True)
        else:
            valid.append(False)
    valid = np.asarray(valid, dtype=bool)
    population = population[valid]
    pop_coeffs = np.asarray(pop_coeffs, dtype=object)
    pop_boot = np.asarray(pop_boot, dtype=int)
    pop_depth = np.asarray(pop_depth, dtype=int)
    if ckpts is not None:
        ckpts = ckpts[valid]

    # finetune polynomial CNNs
    NP = len(population)
    pop_ckpts = []
    pop_accuracy = []
    for i in tqdm(range(NP), desc=prefix+"| Fine-tune"):
        coeffs = pop_coeffs[i].tolist()
        gene = population[i]
        boot = pop_boot[i]
        # model and training setup
        modelx = create_model(args.arch+"_autofhe", args.dataset, coeffs.copy())
        modelx.to(device)
        if ckpts is not None:
            ckpt_path = ckpts[i]
            checkpoint = torch.load(ckpt_path, map_location=device)
            modelx.load_state_dict(checkpoint["ckpt"], strict=False)
        else:
            modelx.load_state_dict(model.state_dict(), strict=False)
        # training setup
        for n, m in modelx.named_parameters():
            if "coef" in n:
                m.requires_grad = False
        optimizer = torch.optim.SGD(modelx.parameters(), args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        # fine-tuning
        acc_best = float("-inf")
        for e in range(args.epochs):
            # training
            train(modelx, model, train_loader, optimizer, num_classes, device, args.grad_clip, args.alpha)
            # validation
            acc = validate(modelx, minival_loader, num_classes, device)
            # early quit
            if acc > acc_best:
                acc_best = acc
            else:
                break
            # estimate range
            if e < args.epochs:
                correct_index = []
                with torch.no_grad():
                    for images, targets, index in train_loader_idx:
                        images = images.to(device)
                        targets = targets.to(device)
                        outputs = modelx(images)
                        logits, outputs = torch.max(outputs, dim=1)
                        correct = torch.logical_and(outputs == targets, logits < 50)
                        correct = correct.cpu()
                        correct_index.extend(index[correct].tolist())
                if len(correct_index) > 0:
                    subloader = create_subloader(args.dataset, args.data, correct_index, args.batch_size, args.num_workers)
                    modelx.enable_track_relu()
                    with torch.no_grad():
                        for images, _ in subloader:
                            images = images.to(device)
                            modelx(images)
                    modelx.disable_track_relu()
        pop_accuracy.append(acc_best)
        pop_ckpts.append(os.path.join(save, f"{i}.ckpt"))
        torch.save({"ckpt": modelx.state_dict(),
                    "coeffs": coeffs,
                    "acc": acc_best,
                    "gene": gene,
                    "boot": boot},
                   os.path.join(save, f"{i}.ckpt"))
    pop_ckpts = np.asarray(pop_ckpts, dtype=object)
    pop_accuracy = np.asarray(pop_accuracy, dtype=float)

    return population, pop_coeffs, pop_accuracy, pop_boot, pop_depth, pop_ckpts


if __name__ == '__main__':
    main()
