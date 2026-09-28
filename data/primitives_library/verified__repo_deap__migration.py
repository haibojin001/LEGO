__all__ = ["migRing"]


def migRing(populations, k, selection, replacement=None, migarray=None):
    """Perform migration among populations arranged in a ring."""
    count = len(populations)

    if migarray is None:
        migarray = list(range(1, count)) + [0]

    outgoing = [[] for _ in range(count)]
    incoming_slots = [[] for _ in range(count)]

    for deme_index in range(count):
        chosen = selection(populations[deme_index], k)
        outgoing[deme_index].extend(chosen)

        if replacement is None:
            incoming_slots[deme_index] = outgoing[deme_index]
        else:
            incoming_slots[deme_index].extend(
                replacement(populations[deme_index], k)
            )

    for source_index, destination_index in enumerate(migarray):
        destination = populations[destination_index]
        for position, individual in enumerate(incoming_slots[destination_index]):
            slot = destination.index(individual)
            destination[slot] = outgoing[source_index][position]