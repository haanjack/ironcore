# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from ironcore.preprocessing.serializer import DataSerializer


def test_token_id_65535_fits_uint16():
    result = DataSerializer._tokens_to_array([0, 65535])
    assert result.dtype == np.uint16
    assert result.tolist() == [0, 65535]


def test_token_id_65536_requires_uint32():
    result = DataSerializer._tokens_to_array([65536])
    assert result.dtype == np.uint32
    assert result.tolist() == [65536]


def test_empty_token_stream_is_rejected():
    with pytest.raises(ValueError, match="empty dataset"):
        DataSerializer._tokens_to_array([])
