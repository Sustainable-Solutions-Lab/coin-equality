"""
Central figure style definitions for COIN_equality publication figures.

All plot scripts should import shared constants from here.

Style rules:
- Grid lines must always render behind data: call ax.set_axisbelow(True)
  after any ax.grid() call, or use zorder=0 on grid elements.
- Use the TAX_EQUITY_COLORS palette consistently across all figures.
"""

TAX_EQUITY_ORDER = [0.0, 0.5, 0.998]

TAX_EQUITY_LABELS = {
    0.0: 'Flat tax',
    0.5: 'Progressive tax',
    0.998: 'Highly progressive tax',
}

TAX_EQUITY_COLORS = {
    0.0: '#2C3E50',
    0.5: '#d62728',
    0.998: '#1f77b4',
}

SWEEP_ORDER = ['eta', 'rho', 'beta', 'all']

SWEEP_TITLES = {
    'eta': 'Elasticity of\nmarginal utility (\u03b7)',
    'rho': 'Effective rate of pure\ntime preference (\u03c1)',
    'beta': 'Damage distribution\nexponent (\u03b2)',
    'all': 'All parameters\n(combined)',
}
