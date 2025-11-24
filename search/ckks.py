import math


def coeffs2ckks(arch: str, dataset: str, coeffs: list):

    if 'resnet' in arch:

        if 'resnet20' in arch:
            end_num = 2
        elif 'resnet32' in arch:
            end_num = 4
        elif 'resnet44' in arch:
            end_num = 6
        elif 'resnet56' in arch:
            end_num = 8
        elif 'resnet110' in arch:
            end_num = 17
        else:
            raise NotImplementedError(f"Unknown: {arch}")

        coeffs = coeffs.copy()
        boots = 0
        depths = 0

        # ConvBN
        level = 16

        # EvoReLU
        coef = coeffs.pop(0)
        depth = int(sum([math.ceil(math.log2(len(c))) for c in coef]))
        if depth > 0:
            depth += 1
        depths += depth
        level -= depth
        if level < 0:
            return False

        boot_loc = []

        for j in range(0, 3):
            for k in range(0, end_num + 1):

                tmp = level

                # ConvBN
                level -= 2
                if level < 0:
                    return False

                # Possible Bootstrapping
                coef = coeffs.pop(0)
                depth = int(sum([math.ceil(math.log2(len(c))) for c in coef]))
                if depth > 0:
                    depth += 1
                depths += depth
                if level < depth:
                    boots += 1
                    level = 16
                    boot_loc.append(1)
                else:
                    boot_loc.append(0)

                # EvoReLU
                level -= depth
                if level < 0:
                    return False

                # Possible Bootstrapping
                if level < 2:
                    boots += 1
                    level = 16
                    boot_loc.append(1)
                else:
                    boot_loc.append(0)

                # ConvBN
                level -= 2
                if level < 0:
                    return False

                # Residual
                if j >= 1 and k == 0:
                    tmp = tmp - 1 if dataset == 'cifar10' else tmp - 2
                else:
                    tmp = tmp - 1
                level = min(level, tmp)
                if level < 0:
                    return False

                # Possible Bootstrapping
                coef = coeffs.pop(0)
                depth = int(sum([math.ceil(math.log2(len(c))) for c in coef]))
                if depth > 0:
                    depth += 1
                depths += depth
                if level < depth:
                    boots += 1
                    level = 16
                    boot_loc.append(1)
                else:
                    boot_loc.append(0)

                # EvoReLU
                level -= depth
                if level < 0:
                    return False

                # Possible Bootstrapping
                if level < 2:
                    boots += 1
                    level = 16
                    boot_loc.append(1)
                else:
                    boot_loc.append(0)

        # AvgPooling
        level -= 1
        if level < 0:
            return False

        # Fully-Connected
        level -= 1
        if level < 0:
            return False

        return boots, depths, boot_loc

    elif "vgg" in arch:
        if 'vgg11' in arch:
            end_num = 2
        else:
            raise NotImplementedError(f"Unknown: {arch}")

        coeffs = coeffs.copy()
        boots = 0
        depths = 0

        # ConvBN
        level = 16

        # EvoReLU
        coef = coeffs.pop(0)
        depth = int(sum([math.ceil(math.log2(len(c))) for c in coef]))
        if depth > 0:
            depth += 1
        depths += depth
        level -= depth
        if level < 0:
            return False

        boot_loc = []

        for j in range(0, 3):
            for k in range(0, end_num + 1):

                # ConvBN
                level -= 2
                if level < 0:
                    return False

                # Possible Bootstrapping
                coef = coeffs.pop(0)
                depth = int(sum([math.ceil(math.log2(len(c))) for c in coef]))
                if depth > 0:
                    depth += 1
                depths += depth
                if level < depth:
                    boots += 1
                    level = 16
                    boot_loc.append(1)
                else:
                    boot_loc.append(0)

                # EvoReLU
                level -= depth
                if level < 0:
                    return False

                # Possible Bootstrapping
                if level < 2:
                    boots += 1
                    level = 16
                    boot_loc.append(1)
                else:
                    boot_loc.append(0)

        # AvgPooling
        level -= 1
        if level < 0:
            return False

        # Fully-Connected
        level -= 1
        if level < 0:
            return False

        return boots, depths, boot_loc

    else:

        raise NotImplementedError(f"=> Unknown architecture: {arch}")