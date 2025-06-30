import pandas as pd

class GenerateChurnLabels:
    def __init__(self, inactivity_threshold=90):
        """
        Initializes the churn label generator with a specified inactivity threshold in days.
        param: 
            inactivity_threshold: Number of days of inactivity to consider for churn.
        """
        self.inactivity_threshold = inactivity_threshold

    def fit(self, df):
        return self

    def transform(self, df):
        """
        Fits the churn label generator to the DataFrame.
        param 
            df: DataFrame assuming it has columns: 'atm_transfer_in', 'atm_transfer_out', 'bank_transfer_in',
            'bank_transfer_out', 'crypto_in', 'crypto_out', 'bank_transfer_in_volume',
            'bank_transfer_out_volume', 'crypto_in_volume', 'crypto_out_volume', 'complaints',
            'touchpoints', 'csat_scores'
        return: DataFrame with churn labels added.
        """
        df = self.__generate_activity_flag(df)
        df = self.__generate_churn_labels(df)
        return df
    
    # Generate activity flag based on available data columns
    def __generate_activity_flag(self, df):
        """
        Adds a binary 'activity_flag' column to the dataframe. 1 if any activity exists, else 0.
        """
        activity_columns = [
            'atm_transfer_in',
            'atm_transfer_out',
            'bank_transfer_in',
            'bank_transfer_out',
            'crypto_in',
            'crypto_out',
            'bank_transfer_in_volume',
            'bank_transfer_out_volume',
            'crypto_in_volume',
            'crypto_out_volume',
            'complaints',
            'touchpoints',
            'csat_scores'
        ]

        def has_activity(row):
            for col in activity_columns:
                val = row.get(col)
                if isinstance(val, (int, float)) and val > 0:
                    return 1
                elif isinstance(val, list) and len(val) > 0:
                    return 1
                elif isinstance(val, dict) and len(val) > 0:
                    return 1
            return 0

        df['activity_flag'] = df.apply(has_activity, axis=1)
        return df

    # generate churn labels based on inactivity
    def __generate_churn_labels(self, df):
        """
        Adds a 'churn_label' column to the DataFrame based on 90 days of inactivity.
        Assumes `df` has columns: ['customer_id', 'date', 'activity_flag'].
        """
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['customer_id', 'date'])

        churn_labels = []
        customer_groups = df.groupby('customer_id') # these groups are tuples of (customer_id, group_df)

        for customer_id, group in customer_groups:
            group = group.sort_values('date') # each group is a dataframe for a single customer
            dates = group['date'].tolist()
            flags = group['activity_flag'].tolist()
            churned = [0] * len(group)

            data_cutoff = df['date'].max()  # Last known date in dataset

            for i in range(len(group)):
                ref_date = dates[i]
                window_end = ref_date + pd.Timedelta(days=90)

                if window_end > data_cutoff:
                    churned[i] = None  # Not enough future data to assign label
                    continue

                activity_within_window = any(
                    flags[j] == 1 and dates[j] >= ref_date and dates[j] <= window_end
                    for j in range(i + 1, len(group))
                )
                churned[i] = 0 if activity_within_window else 1

            group = group.copy()
            group['churn_label'] = churned
            churn_labels.append(group)

        result_df = pd.concat(churn_labels, ignore_index=True)
        return result_df

