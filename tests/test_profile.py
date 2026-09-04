import pytest

from tileops.perf.profile import get_profile_path, load_profile


class TestGetProfilePath:
    @pytest.mark.smoke
    def test_ascend910b1_exists(self):
        path = get_profile_path("ascend910b1")
        assert path.exists()
        assert path.suffix == ".yaml"

    @pytest.mark.smoke
    def test_unknown_device_raises(self):
        with pytest.raises(FileNotFoundError):
            get_profile_path("nonexistent_device")


class TestLoadProfile:
    @pytest.mark.smoke
    def test_ascend910b1_top_level_keys(self):
        profile = load_profile("ascend910b1")
        assert profile["device"] == "Ascend910B1"
        assert profile["soc"] == "ascend910b1"
        assert "hbm" in profile
        # The AI Core's two units, which are what the roof keys name.
        assert "cube" in profile
        assert "vector" in profile
