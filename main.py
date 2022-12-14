from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
import pandas as pd
import time
from imblearn.ensemble import BalancedRandomForestClassifier

# Start variables
start = time.time()
seed = 123
n_splits = 5

path = './dataset.csv'
df = pd.read_csv(path, sep=',')

df = df.drop('id', axis=1)

y = df.ageingRelated
X = df.drop('ageingRelated', axis=1)

# Create classifier
clf = BalancedRandomForestClassifier(random_state=seed, class_weight='balanced_subsample')

# Calculate metrics
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
scoring = ['precision', 'precision_micro', 'precision_macro', 'precision_weighted', 'recall', 'recall_micro',
           'recall_macro', 'recall_weighted', 'f1', 'f1_micro', 'f1_macro', 'f1_weighted', 'accuracy', 'roc_auc']
scores = cross_validate(clf, X, y, cv=cv, scoring=scoring)

# Build output file
output = 'Métrica,,,,,,Média,DesvioPadrão\n'
for score in scores:
    print(f'{score}: {scores[score].mean():.3f} \nDeviation: {scores[score].std():.3f}\n{scores[score]}')
    values = ''
    for value in scores[score]:
        values += f'{value:.3f},'
    output += f'{score.replace("test_", "")},{values}{scores[score].mean():.3f},{scores[score].std():.3f}\n'

end = time.time()
print('\nElapsed Time: %.2f' % (end - start))

# Save output
f = open("output.csv", "a")
f.write(output)
f.close()
