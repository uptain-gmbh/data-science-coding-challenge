import re
import datetime
import pandas


class Preprocess:

    @classmethod
    def extract_year(cls, data: str) -> int:
        """
        Extracts year from data using regex
        :returns: year
        """
        year_match = re.search(r'\d{4}', data)
        if year_match:
            year = int(year_match.group())
            if 1924 <= year <= 2024:
                return year

        year_match = re.search(r'\d{2}', data)
        if year_match:
            year = int(year_match.group())
            if 1 <= year <= 24:
                return 2000 + year
            elif 25 <= year <= 99:
                return 1900 + year

        return 0

    @classmethod
    def extract_domain(cls, data: str):
        """
        Fetches Domain
        """
        domain = data.split('@')[-1]
        return domain.split('.')[0]

    @classmethod
    def age_class(cls, year: int):
        """
        returns label based on year
        """
        if year > 0:
            current_year = datetime.datetime.now().year
            age_from_year = current_year - year
            if age_from_year < 18:
                return 'unsure'
            elif 18 <= age_from_year <= 30:
                return 'young'
            elif 30 < age_from_year <= 50:
                return 'medium'
            else:
                return 'old'
        return 'unsure'


    def run(self, data):
        features = dict()
        features['year'] = self.extract_year(data)
        features['domain'] = self.extract_domain(data)
        return features

    def label(self, df: pandas.DataFrame):
        """
        Labels data
        :returns labeled data:
        """
        df['age_class'] = df.apply(lambda row: self.age_class(row['year']), axis=1)
        return df
