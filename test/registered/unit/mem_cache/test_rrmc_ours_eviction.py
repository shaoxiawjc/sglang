from types import SimpleNamespace

from sglang.srt.mem_cache.mamba_radix_cache import TreeNode
from sglang.srt.mem_cache.rrmc_ours_eviction import RRMCOursEvictionPolicy


class _Cache:
    def __init__(self):
        self.root_node = TreeNode(0)
        self.req_to_token_pool = SimpleNamespace(mamba_pool=None)
        self.token_to_kv_pool_allocator = None

    def _ours_is_full_candidate(self, node):
        return node.value is not None

    def _ours_is_mamba_candidate(self, node):
        return node.mamba_value is not None

    def _ours_is_host_full_candidate(self, node):
        return node.host_value is not None

    def _ours_is_host_mamba_candidate(self, node):
        return node.mamba_host_value is not None


def _node(node_id: int, parent: TreeNode, prefix_len: int) -> TreeNode:
    node = TreeNode(node_id)
    node.parent = parent
    node.rrmc_prefix_tokens = prefix_len
    node.value = [node_id]
    parent.children[node_id] = node
    return node


def _policy(alpha: float) -> RRMCOursEvictionPolicy:
    policy = RRMCOursEvictionPolicy(
        cache=_Cache(),
        alpha=alpha,
        debug=False,
        model_config={"hidden_size": 1, "num_hidden_layers": 1},
    )
    policy.cost_estimator.node_memory_bytes = lambda node, action: 1.0
    return policy


def test_equal_efficiency_selects_oldest_node():
    policy = _policy(alpha=0.5)
    root = policy.cache.root_node
    nodes = [_node(1, root, 8), _node(2, root, 16), _node(3, root, 24)]
    policy.cost_estimator.saved_flops = lambda prefix_len: 1.0

    for node in nodes:
        policy.register_node(node)

    assert policy.select_full_candidate() is nodes[0]


def test_efficiency_score_can_retain_old_high_value_node():
    policy = _policy(alpha=1.0)
    root = policy.cache.root_node
    old_high_eff = _node(1, root, 100)
    middle_low_eff = _node(2, root, 1)
    newest_mid_eff = _node(3, root, 50)

    for node in [old_high_eff, middle_low_eff, newest_mid_eff]:
        policy.register_node(node)

    assert policy.select_full_candidate() is middle_low_eff


def test_selection_does_not_depend_on_candidate_set_insertion_order():
    policy = _policy(alpha=0.0)
    root = policy.cache.root_node
    newest = _node(1, root, 1)
    oldest = _node(2, root, 1)
    middle = _node(3, root, 1)
    policy.cost_estimator.saved_flops = lambda prefix_len: 1.0

    for node in [oldest, middle, newest]:
        policy.register_node(node)
    policy.full_candidates.clear()
    for node in [newest, middle, oldest]:
        policy.full_candidates.add(node.id)

    assert policy.select_full_candidate() is oldest
