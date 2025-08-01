#!/usr/bin/env python3
"""Check if scipy is available for the new metrics calculator features."""

try:
    from scipy.stats import ttest_ind, mannwhitneyu
    print("✅ scipy.stats import successful")
    
    # Test basic functionality
    import numpy as np
    data1 = np.array([1, 2, 3, 4, 5])
    data2 = np.array([2, 3, 4, 5, 6])
    
    t_stat, p_val = ttest_ind(data1, data2)
    print(f"✅ t-test successful: t={t_stat:.3f}, p={p_val:.3f}")
    
    u_stat, p_val_mw = mannwhitneyu(data1, data2)
    print(f"✅ Mann-Whitney U test successful: U={u_stat:.3f}, p={p_val_mw:.3f}")
    
    print("✅ ALL SCIPY TESTS PASSED")
    
except ImportError as e:
    print(f"❌ scipy import failed: {e}")
    print("💡 You may need to install scipy: pip install scipy")
except Exception as e:
    print(f"❌ scipy test failed: {e}")
