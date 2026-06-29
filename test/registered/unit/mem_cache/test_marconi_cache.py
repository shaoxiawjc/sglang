import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.mem_cache.mamba_radix_cache import LRUList, TreeNode
from sglang.srt.mem_cache.marconi_cache import MarconiCache
from sglang.srt.mem_cache.radix_cache import (
    RadixKey,
    _key_match_page_size1,
    get_child_key,
)


class TestMarconiCacheKVOwnership(unittest.TestCase):
    def setUp(self):
        self.cache = object.__new__(MarconiCache)
        self.cache.root_node = TreeNode()
        self.cache.root_node.key = RadixKey([], None)
        self.cache.root_node.value = []
        self.cache.root_node.marconi_prefix_tokens = 0
        self.cache.get_child_key_fn = get_child_key
        self.cache.key_match_fn = _key_match_page_size1
        self.cache.full_lru_list = LRUList(mamba=False)
        self.cache.mamba_lru_list = LRUList(mamba=True)
        self.cache.full_evictable_size_ = 0
        self.cache.token_to_kv_pool_allocator = Mock()
        self.cache._on_token_node_created = Mock()

        self.key = RadixKey([1, 2], None)
        self.cached_indices = torch.tensor([10, 11], dtype=torch.int32)
        self.cache._insert_kv_only(
            key=self.key,
            value=self.cached_indices,
            duplicate_free_from=0,
            forced_splits=set(),
        )
        self.cache.token_to_kv_pool_allocator.free.reset_mock()

    def test_reinserting_tree_owned_indices_does_not_free_them(self):
        self.cache._insert_kv_only(
            key=self.key,
            value=self.cached_indices.clone(),
            duplicate_free_from=0,
            forced_splits=set(),
        )

        self.cache.token_to_kv_pool_allocator.free.assert_not_called()
        self.assertEqual(self.cache.full_evictable_size_, 2)

    def test_reinserting_new_indices_frees_duplicates(self):
        incoming = torch.tensor([20, 21], dtype=torch.int32)

        self.cache._insert_kv_only(
            key=self.key,
            value=incoming,
            duplicate_free_from=0,
            forced_splits=set(),
        )

        freed = self.cache.token_to_kv_pool_allocator.free.call_args.args[0]
        self.assertTrue(torch.equal(freed, incoming))

    def test_mixed_shared_and_new_indices_frees_only_new_indices(self):
        incoming = torch.tensor([10, 21], dtype=torch.int32)

        self.cache._insert_kv_only(
            key=self.key,
            value=incoming,
            duplicate_free_from=0,
            forced_splits=set(),
        )

        freed = self.cache.token_to_kv_pool_allocator.free.call_args.args[0]
        self.assertTrue(torch.equal(freed, torch.tensor([21], dtype=torch.int32)))

    def test_protected_prefix_is_not_freed(self):
        incoming = torch.tensor([20, 21], dtype=torch.int32)

        self.cache._insert_kv_only(
            key=self.key,
            value=incoming,
            duplicate_free_from=1,
            forced_splits=set(),
        )

        freed = self.cache.token_to_kv_pool_allocator.free.call_args.args[0]
        self.assertTrue(torch.equal(freed, torch.tensor([21], dtype=torch.int32)))

    def test_forced_split_does_not_free_tree_owned_indices(self):
        key = RadixKey([3, 4, 5, 6], None)
        indices = torch.tensor([30, 31, 32, 33], dtype=torch.int32)
        self.cache._insert_kv_only(
            key=key,
            value=indices,
            duplicate_free_from=0,
            forced_splits=set(),
        )
        self.cache.token_to_kv_pool_allocator.free.reset_mock()

        self.cache._insert_kv_only(
            key=key,
            value=indices.clone(),
            duplicate_free_from=0,
            forced_splits={2},
        )

        self.cache.token_to_kv_pool_allocator.free.assert_not_called()
        self.assertEqual(self.cache.full_evictable_size_, 6)


class TestMarconiCachePageAlignment(unittest.TestCase):
    def test_unfinished_short_input_is_not_inserted_into_paged_cache(self):
        cache = object.__new__(MarconiCache)
        cache.disable = False
        cache.page_size = 256
        cache._skip_cache_unfinished_req = Mock()
        cache._release_unattached_marconi_slots = Mock()
        req = SimpleNamespace(
            fill_ids=list(range(80)), origin_input_ids=list(range(80))
        )

        cache.cache_unfinished_req(req)

        cache._skip_cache_unfinished_req.assert_called_once_with(req, 80)
        cache._release_unattached_marconi_slots.assert_called_once_with(req)

    def test_unfinished_cache_length_is_capped_to_input_and_page_aligned(self):
        cache = object.__new__(MarconiCache)
        cache.page_size = 256

        self.assertEqual(cache._align_down(min(600, 530)), 512)

    def test_unfinished_only_inserts_through_reached_checkpoint(self):
        cache = object.__new__(MarconiCache)
        cache.disable = False
        cache.page_size = 2
        cache.root_node = TreeNode()
        checkpoint_node = TreeNode()
        canonical_indices = torch.tensor([100, 101, 102, 103], dtype=torch.int64)
        cache.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([[10, 11, 12, 13, 14, 15]], dtype=torch.int64),
            write=Mock(),
        )
        cache._cache_input_kv_path = Mock(return_value=(4, checkpoint_node))
        cache._collect_prefix_indices = Mock(return_value=canonical_indices)
        cache.dec_lock_ref = Mock()
        cache.inc_lock_ref = Mock()
        cache._release_unattached_marconi_slots = Mock()
        req = SimpleNamespace(
            fill_ids=[1, 2, 3, 4, 5, 6],
            origin_input_ids=[1, 2, 3, 4, 5, 6],
            req_pool_idx=0,
            cache_protected_len=0,
            last_node=cache.root_node,
            extra_key=None,
            _marconi_admission_seqlen=4,
        )

        cache.cache_unfinished_req(req)

        call = cache._cache_input_kv_path.call_args.kwargs
        self.assertEqual(call["token_ids"], [1, 2, 3, 4])
        self.assertTrue(
            torch.equal(call["kv_indices"], torch.tensor([10, 11, 12, 13]))
        )
        self.assertTrue(
            torch.equal(
                req.prefix_indices,
                torch.tensor([100, 101, 102, 103, 14, 15]),
            )
        )
        self.assertEqual(req.cache_protected_len, 4)
        self.assertIs(req.last_node, checkpoint_node)

    def test_unfinished_without_reached_checkpoint_does_not_insert(self):
        cache = object.__new__(MarconiCache)
        cache.disable = False
        cache.page_size = 2
        cache._cache_input_kv_path = Mock()
        cache._skip_cache_unfinished_req = Mock()
        cache._release_unattached_marconi_slots = Mock()
        req = SimpleNamespace(
            fill_ids=[1, 2, 3, 4],
            origin_input_ids=[1, 2, 3, 4, 5, 6],
            cache_protected_len=0,
            _marconi_admission_seqlen=6,
        )

        cache.cache_unfinished_req(req)

        cache._cache_input_kv_path.assert_not_called()
        cache._skip_cache_unfinished_req.assert_called_once_with(req, 4)

    def test_unfinished_preserves_unaligned_request_tail(self):
        cache = object.__new__(MarconiCache)
        cache.disable = False
        cache.page_size = 2
        cache.root_node = TreeNode()
        new_last_node = TreeNode()
        canonical_indices = torch.tensor([100, 101], dtype=torch.int64)
        cache.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([[10, 11, 12]], dtype=torch.int64),
            write=Mock(),
        )
        cache._cache_input_kv_path = Mock(return_value=(2, new_last_node))
        cache._collect_prefix_indices = Mock(return_value=canonical_indices)
        cache.dec_lock_ref = Mock()
        cache.inc_lock_ref = Mock()
        cache._release_unattached_marconi_slots = Mock()
        req = SimpleNamespace(
            fill_ids=[1, 2, 3],
            origin_input_ids=[1, 2, 3],
            req_pool_idx=0,
            cache_protected_len=0,
            last_node=cache.root_node,
            extra_key=None,
            _marconi_admission_seqlen=2,
        )

        cache.cache_unfinished_req(req)

        self.assertTrue(
            torch.equal(
                req.prefix_indices,
                torch.tensor([100, 101, 12], dtype=torch.int64),
            )
        )
        self.assertEqual(req.cache_protected_len, 2)
        self.assertIs(req.last_node, new_last_node)


class TestMarconiCacheMetrics(unittest.TestCase):
    def test_accepted_prefix_tokens_update_generic_and_marconi_counters(self):
        cache = object.__new__(MarconiCache)
        cache._reset_cache_perf_counters()
        cache.root_node = TreeNode()
        cache.root_node.value = []

        cache.record_accepted_hit_tokens(256)

        metrics = cache.get_cache_metrics()
        self.assertEqual(metrics["total_accepted_hit_tokens"], 256)
        self.assertEqual(metrics["total_marconi_accepted_prefix_tokens"], 256)

    def test_non_positive_accepted_prefix_tokens_are_ignored(self):
        cache = object.__new__(MarconiCache)
        cache._reset_cache_perf_counters()

        cache.record_accepted_hit_tokens(0)

        self.assertEqual(cache.total_accepted_hit_tokens, 0)
        self.assertEqual(cache.total_marconi_accepted_prefix_tokens, 0)


if __name__ == "__main__":
    unittest.main()
