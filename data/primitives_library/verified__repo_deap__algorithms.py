import random

from . import tools


def varAnd(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(individual) for individual in population]

    for index in range(1, len(offspring), 2):
        if random.random() < cxpb:
            first, second = toolbox.mate(offspring[index - 1], offspring[index])
            offspring[index - 1] = first
            offspring[index] = second
            del offspring[index - 1].fitness.values
            del offspring[index].fitness.values

    for index, individual in enumerate(offspring):
        if random.random() < mutpb:
            offspring[index], = toolbox.mutate(individual)
            del offspring[index].fitness.values

    return offspring


def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    unevaluated = [individual for individual in population
                   if not individual.fitness.valid]
    results = toolbox.map(toolbox.evaluate, unevaluated)
    for individual, values in zip(unevaluated, results):
        individual.fitness.values = values

    if halloffame is not None:
        halloffame.update(population)

    values = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(unevaluated), **values)
    if verbose:
        print(logbook.stream)

    for generation in range(1, ngen + 1):
        selected = toolbox.select(population, len(population))
        offspring = varAnd(selected, toolbox, cxpb, mutpb)

        unevaluated = [individual for individual in offspring
                       if not individual.fitness.valid]
        results = toolbox.map(toolbox.evaluate, unevaluated)
        for individual, values in zip(unevaluated, results):
            individual.fitness.values = values

        if halloffame is not None:
            halloffame.update(offspring)

        population[:] = offspring

        values = stats.compile(population) if stats else {}
        logbook.record(gen=generation, nevals=len(unevaluated), **values)
        if verbose:
            print(logbook.stream)

    return population, logbook


def varOr(population, toolbox, lambda_, cxpb, mutpb):
    assert cxpb + mutpb <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0."
    )

    offspring = []
    for _ in range(lambda_):
        draw = random.random()

        if draw < cxpb:
            parent1, parent2 = [
                toolbox.clone(individual)
                for individual in random.sample(population, 2)
            ]
            child1, child2 = toolbox.mate(parent1, parent2)
            del child1.fitness.values
            offspring.append(child1)

        elif draw < cxpb + mutpb:
            parent = toolbox.clone(random.choice(population))
            child, = toolbox.mutate(parent)
            del child.fitness.values
            offspring.append(child)

        else:
            offspring.append(random.choice(population))

    return offspring


def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    unevaluated = [individual for individual in population
                   if not individual.fitness.valid]
    results = toolbox.map(toolbox.evaluate, unevaluated)
    for individual, values in zip(unevaluated, results):
        individual.fitness.values = values

    if halloffame is not None:
        halloffame.update(population)

    values = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(unevaluated), **values)
    if verbose:
        print(logbook.stream)

    for generation in range(1, ngen + 1):
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        unevaluated = [individual for individual in offspring
                       if not individual.fitness.valid]
        results = toolbox.map(toolbox.evaluate, unevaluated)
        for individual, values in zip(unevaluated, results):
            individual.fitness.values = values

        if halloffame is not None:
            halloffame.update(offspring)

        population[:] = toolbox.select(population + offspring, mu)

        values = stats.compile(population) if stats else {}
        logbook.record(gen=generation, nevals=len(unevaluated), **values)
        if verbose:
            print(logbook.stream)

    return population, logbook


def eaMuCommaLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                    stats=None, halloffame=None, verbose=__debug__):
    assert lambda_ >= mu, (
        "lambda must be greater or equal to mu."
    )

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    unevaluated = [individual for individual in population
                   if not individual.fitness.valid]
    results = toolbox.map(toolbox.evaluate, unevaluated)
    for individual, values in zip(unevaluated, results):
        individual.fitness.values = values

    if halloffame is not None:
        halloffame.update(population)

    values = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(unevaluated), **values)
    if verbose:
        print(logbook.stream)

    for generation in range(1, ngen + 1):
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        unevaluated = [individual for individual in offspring
                       if not individual.fitness.valid]
        results = toolbox.map(toolbox.evaluate, unevaluated)
        for individual, values in zip(unevaluated, results):
            individual.fitness.values = values

        if halloffame is not None:
            halloffame.update(offspring)

        population[:] = toolbox.select(offspring, mu)

        values = stats.compile(population) if stats else {}
        logbook.record(gen=generation, nevals=len(unevaluated), **values)
        if verbose:
            print(logbook.stream)

    return population, logbook


def eaGenerateUpdate(toolbox, ngen, halloffame=None, stats=None,
                     verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    for generation in range(ngen):
        population = toolbox.generate()

        results = toolbox.map(toolbox.evaluate, population)
        for individual, values in zip(population, results):
            individual.fitness.values = values

        if halloffame is not None:
            halloffame.update(population)

        toolbox.update(population)

        values = stats.compile(population) if stats else {}
        logbook.record(gen=generation, nevals=len(population), **values)
        if verbose:
            print(logbook.stream)

    return population, logbook