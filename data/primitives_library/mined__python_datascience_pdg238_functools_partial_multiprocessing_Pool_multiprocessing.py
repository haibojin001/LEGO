# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg238::functools.partial+multiprocessing.Pool+multiprocessing.cpu_count
# name: functools_multiprocessing_primitive
# summary: Uses functools.partial, multiprocessing.Pool, multiprocessing.cpu_count, sys.exit across 4 repos
# anchor_symbols: ['functools.partial', 'multiprocessing.Pool', 'multiprocessing.cpu_count', 'sys.exit']
# observed in 4 repos: ['allenai__allennlp', 'rnorm__book_sample', 'stellargraph__stellargraph', 'zama-ai__concrete-ml']...

# --- from rnorm__book_sample::code/chapter2/pi.py::generate_points_parallel ---
def generate_points_parallel(n):
    pool = mp.Pool()
    # we ask each process to generate n//mp.cpu_count() points
    return pool.map(count_inside_point, [n//mp.cpu_count()]*mp.cpu_count())

# --- from zama-ai__concrete-ml::script/make_utils/deactivate_docs_admonitions_for_tests.py::main ---
def main(args):
    """Entry point.

    Args:
        args: a Namespace object
    """
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        res = pool.map(process_file, args.files)
        # Exit 0 if all went well as True == 1
        sys.exit(not all(res))

# --- from rnorm__book_sample::code/chapter2/pi.py::generate_points_parallel_set_seed ---
def generate_points_parallel_set_seed(n):
    pool = mp.Pool() # we can also specify the number of processes by Pool(number)
    # we ask each process to generate n//mp.cpu_count() points
    return pool.map(helper, list(zip([n//mp.cpu_count()]*mp.cpu_count(),range(mp.cpu_count()))))

# --- from zama-ai__concrete-ml::script/doc_utils/fix_double_dollars_issues_with_mdformat.py::main ---
def main(args):
    """Entry point.

    Args:
        args (List[str]): a list of arguments
    """
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        res = pool.map(partial(process_file, args=args), args.files)
        # Exit 0 if all went well as True == 1
        sys.exit(not all(res))

# --- from allenai__allennlp::scripts/py2md.py::main ---
def main():
    opts = parse_args()
    outputs = opts.out if opts.out else [None] * len(opts.modules)
    if len(outputs) != len(opts.modules):
        raise ValueError("Number inputs and outputs should be the same.")
    n_threads = cpu_count()
    errors: int = 0
    if len(opts.modules) > n_threads and opts.out:
        # If writing to files, can process in parallel.
        chunk_size = max([1, int(len(outputs) / n_threads)])
        logger.info("Using %d threads", n_threads)
        with Pool(n_threads) as p:
            for result in p.imap(_py2md_wrapper, zip(opts.modules, outputs), chunk_size):
                if not result:
                    errors += 1
    else:
        # If writing to stdout, need to process sequentially. Otherwise the output
        # could get intertwined.
        for module, out in zip(opts.modules, outputs):
            result = py2md(module, out)
            if not result:
                errors += 1
    logger.info("Processed %d modules", len(opts.modules))
    if errors:
        logger.error("Found %d errors", errors)
        sys.exit(1)

# --- from stellargraph__stellargraph::stellargraph/data/epgm.py::EPGM.to_nx_OLD ---
def to_nx_OLD(
        self,
        graph_id,
        directed=False,
        parallel_processing=True,
        n_jobs=multiprocessing.cpu_count(),
        progress=True,
        chunksize=100,
    ):
        """Convert the graph specified by its graph_id to networkx graph"""

        if (
            graph_id in self.G_nx.keys()
        ):  # if self.G_nx[graph_id] already exists, just return it, otherwise evaluate it
            return self.G_nx[graph_id]
        else:
            print("Converting the EPGM graph {} to NetworkX graph...".format(graph_id))
            if not any([graph_id in g["id"] for g in self.G["graphs"]]):
                raise Exception("Graph with id {} does not exist".format(graph_id))

            # List relevant nodes and edges:
            print("...extracting relevant nodes...", end="")
            nodes = [
                v["id"] for v in self.G["vertices"] if graph_id in v["meta"]["graphs"]
            ]
            print(" ...{} nodes extracted...".format(len(nodes)))
            print("...extracting relevant edges...", end="")
            edges = [
                (e["source"], e["target"])
                for e in self.G["edges"]
                if graph_id in e["meta"]["graphs"]
            ]
            print(" ...{} edges extracted...".format(len(edges)))
            # TODO: implement the case of weighted edges

            # create a graph as dict of lists in the format (node_id: [neighbour nodes])
            print("...building the graph as dict of lists...")
            print(
                "...[parallel_processing: {}, n_jobs: {}, progress_bar: {}]".format(
                    parallel_processing, n_jobs, progress
                )
            )

            if parallel_processing:  # parallel execution
                pool = Pool(processes=n_jobs)
                if progress:
                    n = len(nodes)
                    self.G_nx[graph_id] = []

                    # pbar = ProgressBar(
                    #     widgets=[
                    #         SimpleProgress(
                    #             format="%(value_s)s of %(max_value_s)s nodes processed (%(percentage)3d%%)"
                    #         )
                    #     ],
                    #     maxval=n,
                    # ).start()
                    # _ = [pool.apply_async(partial(node_neighbours, edges=edges), args=(v,),
                    #                       callback=self.G_nx[graph_id].append) for v in nodes]
                    # it seems that appending results using callback works much slower than either pool.map_async or pool.map
                    # while len(self.G_nx[graph_id]) != n:
                    #     pbar.update(len(self.G_nx[graph_id]))
                    #     sleep(1)

                    graph = pool.imap(
                        partial(node_neighbours, edges=edges), nodes, chunksize
                    )  # lazy map
                    # evaluate batches of imap, as the progress bar is being updated:
                    while len(self.G_nx[graph_id]) != n:
                        self.G_nx[graph_id].append(next(graph))
                        # pbar.update(len(self.G_nx[graph_id]))

                    # pbar.finish()

                    self.G_nx[graph_id] = dict(self.G_nx[graph_id])

                else:
                    self.G_nx[graph_id] = dict(
                        pool.map(partial(node_neighbours, edges=edges), nodes)
                    )

                pool.close()
                pool.join()

            else:  # sequential execution
                self.G_nx[graph_id] = {
                    v: [e[1] for e in edges if e[0] == v] for v in nodes
                }  # this works ~2.5x faster (for cora dataset) than the above for loop

            print("...converting the graph to nx format...")
            self.G_nx[graph_id] = nx.from_dict_of_lists(self.G_nx[graph_id])

            if directed:
                self.G_nx[graph_id] = self.G_nx[graph_id].to_directed()
            else:
                self.G_nx[graph_id] = self.G_nx[graph_id].to_undirected()

            return self.G_nx[graph_id]
