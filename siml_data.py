import os
import gzip
import pandas as pd

class Featureset:
    def __init__(self, output='features.csv.gz', src_dir='data', cached=True):
        self.cached = cached
        self.integrated_file = output
        self.files = [os.path.join(src_dir, f) for f in os.listdir(src_dir)]

    def get(self) -> pd.DataFrame:
        if self.cached and os.path.exists(self.integrated_file):
            return pd.read_csv(gzip.GzipFile(self.integrated_file, 'rb'), index_col=0)

        file_to_prefix = lambda s: s.split('.')[0].split('/')[-1]
        df = self.load_file(self.files[0]).add_prefix(file_to_prefix(self.files[0]))

        for f in self.files[1:]:
            df = df.merge(
                self.load_file(f).add_prefix(file_to_prefix(f)),
                how='inner',
                copy=False,
                left_index=True,
                right_index=True,
            )
            if len(df) < 1:
                raise Exception(f"empty dataframe after merging with: {f}")

        if self.cached:
            df.to_csv(self.integrated_file, compression='gzip')
        return df

    @staticmethod
    def load_file(f: str) -> pd.DataFrame:
        gf = gzip.GzipFile(f, 'rb')
        df =  pd.read_csv(gf, index_col=0)
        df.set_index(pd.to_datetime(df.index).to_period('D'), inplace=True)
        df.rename({
            'Otwarcie': 'Open',
            'Najwyzszy': 'High',
            'Najnizszy': 'Low',
            'Zamkniecie': 'Close'}, axis=1, inplace=True)
        return df[['Open', 'High', 'Low', 'Close']]
