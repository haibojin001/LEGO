def _bfs_subgraph(self, start_id, forward=True):
        """
        Private method creates a subgraph in a bfs order.

        The forward parameter specifies whether it is a forward or backward
        traversal.
        """
        if forward:
            get_bfs = self.forw_bfs
            get_nbrs = self.out_nbrs
        else:
            get_bfs = self.back_bfs
            get_nbrs = self.inc_nbrs

        g = Graph()
        bfs_list = get_bfs(start_id)
        for node in bfs_list:
            g.add_node(node)

        for node in bfs_list:
            for nbr_id in get_nbrs(node):
                if forward:
                    g.add_edge(node, nbr_id)
                else:
                    g.add_edge(nbr_id, node)

        return g