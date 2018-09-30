from autodiff import gradients, assign



def find_topo_sort(node_list):
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.

    """
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order

def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    #print("node in dfs ", node)
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)



class GradientDescentOptimizer():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def minimize(self, cost):
        trainable_vars = self.find_trainable_vars(cost)
        # trainable_vars = [node for node in topo_order if node.trainable]
        # trainable_vars = list(reversed(trainable_vars))

        print("trainable_vars ", trainable_vars)

        grad_list = gradients(cost, trainable_vars)

        # print("-------------------------------")
        # print("-------------------------------")
        #
        # for grad in grad_list:
        #     print("\nfor grad ----- ", grad)
        #     topo_order = find_topo_sort([grad])
        #     for node in topo_order:
        #         print("node ", node)
        #
        # print("-------------------------------")
        # print("-------------------------------")

        # topo_order = find_topo_sort([W_grad])
        # for node in topo_order:
        #     print("node ", node)

        assert len(trainable_vars) == len(grad_list)
        train_steps = []
        for var, var_grad in zip(trainable_vars, grad_list):
            train_steps.append(assign(var, var - self.learning_rate * var_grad))

        # print("train steps...")
        # topo_order = find_topo_sort(train_steps)
        # for elt in topo_order:
        #     print("node : ", elt)


        return train_steps

    def find_trainable_vars(self, cost):
        """ finds the trainable vars by doing modified DFS where we explore the
        non-trainable nodes first followed by trainable nodes. Or in other words,
        in the returned list, trainable wts which lies further from the root will be followed by
        trainable wts which are nearer to root. The list of gradients will depend
        on this list and we want to flow the incoming gradient (or apply gradients) to trainable wts
        of greater depth first and then to trainable wts of shallow depth because
        gradient of trainable wts of shallow depth depends on incoming gradient
        and some matrix multiply result of lower wts.
        But gradient of trainable wts of greater depth depends directly on
        wts of shallow depth, so we can not modify the wts of shallow depth before
        wts of greater depth when we update the wts in the training.

        So for e.g. suppose we have following cost function:
        W,  W1 = some trainable wts
        x, labels = ad.placeholder(name = "x"), ad.placeholder(name = "labels")

        matmul = ad.matmul(W, x)
        matmul1 = ad.matmul(W1, matmul)
        cost = ad.reduce_mean( ad.softmax_with_cross_entropy(matmul1, labels) )

        W1 is at lesser depth than W in the cost tree. W lies at bottom. So gradient flow has
        to be done first to W and then to W1.
        So in this case topo_order will be [W, W1]
        """

        visited = set()
        topo_order = []
        self.topo_sort_dfs_m(cost, visited, topo_order)
        return topo_order


    def topo_sort_dfs_m(self, node, visited, topo_order):
        """Post-order DFS"""
        if node in visited:
            return
        visited.add(node)
        for n in self.reorder_nodes(node.inputs):
            self.topo_sort_dfs_m(n, visited, topo_order)
        if node.trainable:
            topo_order.append(node)


    def reorder_nodes(self, nodes):
        """ reorder the nodes for DFS. first accumulate all the non trainable
        variables and then append it to all trainable variables. """
        node_list = [node for node in nodes if not node.trainable]
        trainable_vars = [node for node in nodes if node.trainable]
        node_list.extend(trainable_vars)

        return node_list
