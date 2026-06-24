import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.hi_rrmc_mamba_radix_cache import (
    HiRRMCMambaRadixCache,
)
from sglang.srt.mem_cache.mamba_radix_cache import TreeNode
from sglang.srt.mem_cache.rrmc_mamba_radix_cache import (
    RRMC_RANKED_EVICTION_POLICIES,
    RRMCMambaRadixCache,
)
from sglang.srt.server_args import (
    RRMC_RADIX_EVICTION_POLICY_CHOICES,
    ServerArgs,
)


def make_cache(policy: str, depth_lambda: float = 0.5):
    cache = object.__new__(RRMCMambaRadixCache)
    cache.root_node = TreeNode()
    cache.rrmc_radix_eviction_policy = policy
    cache.depth_aware_evict_lambda = depth_lambda
    return cache


def add_child(parent: TreeNode, *, access_time: float, access_count: int = 1):
    node = TreeNode()
    node.parent = parent
    node.last_access_time = access_time
    node.rrmc_access_count = access_count
    parent.children[node.id] = node
    return node


class TestRRMCEvictionPolicy(unittest.TestCase):
    def test_policy_choices_include_lfu_and_depth_aware(self):
        self.assertIn("lfu", RRMC_RADIX_EVICTION_POLICY_CHOICES)
        self.assertIn("depth_aware", RRMC_RADIX_EVICTION_POLICY_CHOICES)
        self.assertEqual(
            RRMC_RANKED_EVICTION_POLICIES,
            {"lfu", "depth_aware", "ours"},
        )

    def test_lfu_prefers_lower_frequency(self):
        cache = make_cache("lfu")
        older_frequent = add_child(
            cache.root_node, access_time=1.0, access_count=4
        )
        newer_infrequent = add_child(
            cache.root_node, access_time=2.0, access_count=1
        )

        selected = cache._select_ranked_candidate(
            [older_frequent, newer_infrequent], memory_kind="full"
        )

        self.assertIs(selected, newer_infrequent)

    def test_lfu_uses_lru_as_tie_breaker(self):
        cache = make_cache("lfu")
        older = add_child(cache.root_node, access_time=1.0, access_count=2)
        newer = add_child(cache.root_node, access_time=2.0, access_count=2)

        selected = cache._select_ranked_candidate(
            [newer, older], memory_kind="full"
        )

        self.assertIs(selected, older)

    def test_record_path_access_updates_each_reused_node(self):
        cache = make_cache("lfu")
        parent = add_child(cache.root_node, access_time=1.0, access_count=1)
        child = add_child(parent, access_time=2.0, access_count=3)

        cache._record_rrmc_path_access([parent, child])

        self.assertEqual(parent.rrmc_access_count, 2)
        self.assertEqual(child.rrmc_access_count, 4)

    def test_depth_aware_can_prefer_newer_deeper_node(self):
        cache = make_cache("depth_aware", depth_lambda=2.0)
        shallow_old = add_child(cache.root_node, access_time=1.0)
        branch = add_child(cache.root_node, access_time=3.0)
        middle = add_child(branch, access_time=4.0)
        deep_new = add_child(middle, access_time=2.0)

        selected = cache._select_ranked_candidate(
            [shallow_old, deep_new], memory_kind="full"
        )

        self.assertIs(selected, deep_new)

    def test_depth_aware_lambda_zero_matches_lru(self):
        cache = make_cache("depth_aware", depth_lambda=0.0)
        older = add_child(cache.root_node, access_time=1.0)
        branch = add_child(cache.root_node, access_time=3.0)
        newer_deep = add_child(branch, access_time=2.0)

        selected = cache._select_ranked_candidate(
            [newer_deep, older], memory_kind="full"
        )

        self.assertIs(selected, older)

    def test_depth_aware_handles_single_candidate(self):
        cache = make_cache("depth_aware", depth_lambda=1.0)
        node = add_child(cache.root_node, access_time=1.0)

        self.assertIs(
            cache._select_ranked_candidate([node], memory_kind="full"),
            node,
        )

    def test_hirrmc_overrides_all_ranked_eviction_streams(self):
        self.assertIn("_evict_full_ranked", HiRRMCMambaRadixCache.__dict__)
        self.assertIn("_evict_mamba_ranked", HiRRMCMambaRadixCache.__dict__)
        self.assertIn("_evict_mamba_host_ranked", HiRRMCMambaRadixCache.__dict__)
        self.assertIn("_evict_host_full_ranked", HiRRMCMambaRadixCache.__dict__)

    def test_negative_depth_lambda_is_rejected(self):
        args = SimpleNamespace(
            enable_rrmc_radix_cache=True,
            enable_marconi_cache=False,
            rrmc_radix_eviction_policy="depth_aware",
            ours_evict_alpha=0.5,
            depth_aware_evict_lambda=-0.1,
        )

        with self.assertRaisesRegex(ValueError, "should be non-negative"):
            ServerArgs._handle_rrmc_eviction_policy(args)


if __name__ == "__main__":
    unittest.main()
