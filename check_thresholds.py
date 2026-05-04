import pandas as pd

mat = pd.read_csv(r'results\category_skills.csv', index_col=0)
df = pd.read_csv(r'buzzpoints_data.csv')

cat_map = {
    'Fine Arts': 'Fine Arts', 'Science': 'Science', 'History': 'History',
    'Literature': 'Literature', 'Religion': 'Beliefs', 'Mythology': 'Beliefs',
    'Philosophy': 'Thought', 'Social Science': 'Thought',
    'Geography': 'Other', 'Current Events': 'Other', 'Other Academic': 'Other', 'Trash': 'Other',
}
def agg(c):
    c = c.strip()
    for k, v in cat_map.items():
        if c.startswith(k): return v
    return 'Other'

df['cat_agg'] = df['category'].apply(agg)
gets = df[df['won'] == True].groupby(['player_slug', 'cat_agg']).size().unstack(fill_value=0)

thresholds = {'Fine Arts': 15, 'History': 20, 'Literature': 20, 'Science': 20,
              'Beliefs': 10, 'Thought': 10, 'Other': 5}

print(f"{'Category':<15} {'Thresh':>6} {'Qualify':>7} {'JustBelow':>9} {'Med(qual)':>9} {'Max':>5}")
print('-' * 58)
for cat in sorted(thresholds):
    if cat not in gets.columns:
        continue
    col = gets[cat]
    t = thresholds[cat]
    qualifiers = int(mat[cat].notna().sum())
    just_below = int(((col > 0) & (col < t)).sum())
    med = col[col >= t].median() if (col >= t).any() else 0
    mx = int(col.max())
    print(f"{cat:<15} {t:>6} {qualifiers:>7} {just_below:>9} {med:>9.0f} {mx:>5}")

print()
print("JustBelow = players with at least 1 get who don't qualify (excluded by threshold)")
