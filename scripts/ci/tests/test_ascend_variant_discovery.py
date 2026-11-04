from scripts.ci.nightly_device import registered_ascend_ops


def test_multiple_interfaces_are_discovered_once_without_default_key():
    registered = {
        "InstanceNormFwdOp": {
            "ascend_input_stats": object(),
            "ascend_train": object(),
            "ascend_infer": object(),
        },
        "BatchNormFwdOp": {"ascend_train": object(), "ascend_infer": object()},
        "RMSNormFwdOp": {"ascend": object()},
        "CudaOnlyOp": {"cuda": object()},
        "EmptyOp": {},
    }
    assert registered_ascend_ops(registered) == [
        "BatchNormFwdOp",
        "InstanceNormFwdOp",
        "RMSNormFwdOp",
    ]
