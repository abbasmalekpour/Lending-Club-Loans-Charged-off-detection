class SortCountOptim(object):
    """
    The `SortCountOptim` class implements the proposed solution of the top 2% problem.
    An instance of `SortCountOptim` accepts the following parameters:

    :param filename: The CSV file containing the data
    :param top_percent: (Optional) the size of the top portion to be considered for counting labels
    """

    def __init__(self, filename, top_percent=.02):
        from numpy import array
        from pandas import read_csv
        # the main container of data
        self.df = read_csv(filename)
        # size of the top portion
        self.top_percent = top_percent
        # the loss function of the SGDClassifier
        self.loss = 'modified_huber'
        # 1s
        df1 = self.df[self.df['def_new'] == 1]
        # 0s
        df0 = self.df[self.df['def_new'] == 0]
        self.N, self.N_ones = len(self.df), len(df1)
        # features and target columns
        cols = list(self.df.columns)
        target = cols.pop(0)
        # weight factor to be calculated
        self.w = array([1. for _ in cols])
        # mean of 1s := M_1
        self.mean_ones = df1[cols].mean().values
        # mean of 0s := M_0
        self.mean_zeros = df0[cols].mean().values
        # numpy containers
        self.X_all = self.df[cols].values
        self.y_all = self.df[target].values
        # SGDClassifier's accuracy score'
        self.SGDScore = 0

    def separate(self, loss=None):
        """

        :param loss: the loss function of the SGDClassifier (default = 'modified_huber')
        :return: None
        """

        from sklearn.linear_model import SGDClassifier
        if loss is not None:
            self.loss = loss
        # initiate the classifier and fit
        clf = SGDClassifier(loss=self.loss, class_weight='balanced', tol=5e-5, max_iter=60)
        clf.fit(self.X_all, self.y_all)
        # accuracy
        self.SGDScore = clf.score(self.X_all, self.y_all)
        # the normal vector of the hyperplane
        self.normal = clf.coef_[0]

    def score(self, x, w=None):
        """
        Finds the number of 1s appeared on the top `p` portion of the inner product list of points and `w`.
        If `w` is not given it will be computed according to `x`, `M_1`, `M_0`, and `n`.

        :param x: the vector of coefficients- :math: `\alpha=x[0], \beta=x[0], \gamma=x[2]`
        :param w: a suggested weight vector; if None, then it will be computed according to the values of `x`
        :return: the total number of 1s appeared in top `100*p%` of the list
        """

        import numpy as np
        alpha, beta, gamma = x[0], x[1], x[2]
        if w is None:
            self.w = alpha * self.mean_ones + beta * self.mean_zeros + gamma * self.normal
        else:
            self.w = w
        idx = 0
        all_p = []
        for p in self.X_all:
            dt = np.dot(self.w, p)
            all_p.append((dt, self.y_all[idx]))
            idx += 1
        srtd = sorted(all_p, key=lambda t: -t[0])
        slctd = srtd[:int(self.top_percent * self.N) + 1]
        return sum([p[1] for p in slctd])

    def random_search(self, cube=3 * [(-1, 1)], n=30):
        """
        Performs a rudimentary random search to find a candidate for :math:`\alpha, \beta, \gamma`

        :param cube: the ranges for the values of each component
        :param n: number of iterations
        :return: a list of tuples; the first tuple is argmax and max, the second tuple is argmin and min.
        """
        import numpy as np
        dim = len(cube)
        scl = np.array([c[1] - c[0] for c in cube])
        intrcpt = np.array([c[0] for c in cube])
        Xs = np.multiply(scl, np.random.rand(n, dim)) + intrcpt
        mn, mx = None, None
        xmn, xmx = None, None
        for x in Xs:
            vl = self.score(x)
            if (mn is None) or vl < mn:
                mn, xmn = vl, x
            if (mx is None) or vl > mx:
                mx, xmx = vl, x
        return [(xmn, mn), (xmx, mx)]


###########################################################################
###########################################################################
# user site profile
"""
This module implements various part of a solution to the screening project of Toptal technical interview.
The module consists of three main classes:

    - `SiteInfo` which extracts patterns and distributions based on users profiles in the training data,
    - `NaiveBayes` is a customized implementation of the Naive Bays method that employs `SiteInfo`s extracts
    - `CatchJoe` that uses the above two classes to classify the tst data as Joe vs. Not Joe.
    
The module has minimum number of dependencies. `Numpy`, `Pandas`, and `matplotlib` are the essentials of any basic
data related project, implemented in python. This module also uses `dateutil` to convert GMT date-times to local ones. 
"""


class SiteInfo(object):
    """
    The `SiteInfo` class takes a DataFrame which supposedly is a subset of training data and generate some features
    such as distributions based on users' profiles. The followings are the parameters of the class:

    :param joe_id: (default=0) the id of the user to be processed by the instance of the class
    :param min_visits: (default=10) the minimum number of times that a website needs to be visited to be considered as
        frequently visited
    :param time_intervals: (default=50) number of sub-intervals for time spent on each site
    :param min_p: minimum value of the probability replacing 0.
    :param df: the pandas DataFrame to be used for feature extraction. If df is None, then the contents
        of `dataset.json` will be used.
    """

    def __init__(self, joe_id=0, min_visits=10, time_intervals=50, min_p=1e-5, df=None):
        from numpy import full
        from datetime import datetime
        from time import strptime
        from pandas import read_json
        self.joe_id = joe_id
        self.time_intervals = time_intervals
        self.min_visits = min_visits
        self.min_p = min_p
        self.min_p_array = full(self.time_intervals, self.min_p)
        if df is None:
            self.df = read_json('dataset.json')
            self.df['AdjDateTime'] = self.df.apply(
                lambda x: utc_to_tz(datetime.strptime("%d-%d-%d %s" % (x['date'].year,
                                                                       x['date'].month,
                                                                       x['date'].day,
                                                                       x['time']),
                                                      "%Y-%m-%d %H:%M:%S"), x['location']), axis=1)
            self.df['numeric_time'] = self.df.apply(lambda x: strptime(str(x['AdjDateTime'].time()),
                                                                       "%H:%M:%S")[3] +
                                                              strptime(str(x['AdjDateTime'].time()),
                                                                       "%H:%M:%S")[4] / 100., axis=1)
        else:
            self.df = df
        self.joe_df = self.df[self.df['user_id'] == self.joe_id]
        joe_browser = set(self.joe_df['browser'])
        joe_os = set(self.joe_df['os'])
        joe_locale = set(self.joe_df['locale'])
        self.same_df = self.df[(self.df['browser'].isin(joe_browser)) &
                               (self.df['os'].isin(joe_os)) &
                               (self.df['locale'].isin(joe_locale)) &
                               (self.df['gender'] == 'm')]
        self.same_id = set(self.same_df['user_id'])

    def profile(self, user_id=None, browser=None, os=None, same=True, exclude=None):
        """
        Extracts time distribution of of visits of all frequently visited websites.

        :param user_id: (default=None) a list of users to be combined together to produce a single profile.
            If None, all users will be considered
        :param browser: (default=None) if set, the dataset will be filtered by browser
        :param os: (default=None) if set, the dataset will be filtered by operating system
        :param same: (default=True) if True, the dataset will be filtered by similar users, i.e., those users
            with same gender, locale, os and browsers.
        :param exclude: (default=None) the user to be excluded from calculations.
        :return: average length and actual lengths of usage of each site in the final filtered data
        """
        sites = set()
        lengths = {}
        avg_length = {}
        if same:
            loc_df = self.same_df
        else:
            loc_df = self.df
        if user_id is not None:
            loc_df = loc_df[loc_df['user_id'] == user_id]
        if exclude is not None:
            loc_df = loc_df[loc_df['user_id'] != user_id]

        if browser is not None:
            loc_df = loc_df[loc_df['browser'] == browser]
        if os is not None:
            loc_df = loc_df[loc_df['os'] == os]
        for idx, row in loc_df.iterrows():
            for itm in row['sites']:
                sites.add(itm['site'])
                if itm['site'] not in lengths:
                    lengths[itm['site']] = [itm['length']]
                else:
                    lengths[itm['site']] += [itm['length']]
        for site in sites:
            avg_length[site] = (sum(lengths[site]) / len(lengths[site]), len(lengths[site]))
        return avg_length, lengths

    def find_top(self, user_id=None, same=True, exclude=None):
        """
        Finds top most visited sites by a group of users or a single user

        :param user_id: (default=None) a list of users or a single user whose site visits to be analysed. If None
            the `joe_id` will be used
        :param same: (default=True) if True, the DataFrame will be filtered just to include similar users
        :param exclude: users to be excluded
        :return: a dictionary of top averages and corresponding top visit lengths
        """
        avg_length, length = self.profile(user_id=user_id, same=same, exclude=exclude)
        top_avgs, top_lengths = {}, {}
        for site in avg_length:
            if avg_length[site][1] >= self.min_visits:
                top_avgs[site] = avg_length[site]
                top_lengths[site] = length[site]
        return top_avgs, top_lengths

    def site_usage_histo(self, user_id=None, same=True, exclude=None):
        """
        Plots and saves the visit time distribution of each site

        :param user_id: (default=None) a list of users or a single user whose site visits to be analysed. If None
            the `joe_id` will be used
        :param same: (default=True) if True, the DataFrame will be filtered just to include similar users
        :param exclude: users to be excluded
        :return: None
        """
        from numpy import array, histogram, exp
        from matplotlib import pyplot as plt
        plt.figure(figsize=(10, 16))
        num_bins = self.time_intervals
        _, lngth = self.find_top(user_id=user_id, same=same, exclude=exclude)
        n = len(lngth)
        idx = 1
        for s in lngth:
            plt.subplot(n * 100 + 10 + idx)
            X = array(lngth[s])
            y_, x_ = histogram(X, num_bins)
            y_ = y_ / sum(y_)
            plt.plot(x_[1:], y_)
            My = y_.max()
            mx = x_.min()
            Mx = x_.max()
            pdf = lambda z, My=My, Mx=Mx, mx=mx: My * exp(-My * self.time_intervals * (z - mx) / (Mx - mx))
            plt.legend([s])
            plt.plot(x_, pdf(x_))
            idx += 1

    def site_usage_prob(self, user_id=None, same=True, exclude=None):
        """
        Computes the visit time distribution of each site

        :param user_id: (default=None) a list of users or a single user whose site visits to be analysed. If None
            the `joe_id` will be used
        :param same: (default=True) if True, the DataFrame will be filtered just to include similar users
        :param exclude: users to be excluded
        :return: a dictionary with sites as keys and a triple of time intervals, corresponding values and
            their continuous approximation
        """
        from numpy import array, histogram, maximum, exp
        num_bins = self.time_intervals
        _, lngth = self.find_top(user_id=user_id, same=same, exclude=exclude)
        probs = {}
        for s in lngth:
            X = array(lngth[s])
            y_, x_ = histogram(X, num_bins)
            y_ = y_ / sum(y_)
            My = y_.max()
            mx = x_.min()
            Mx = x_.max()
            pdf = lambda z, My=My, Mx=Mx, mx=mx: My * exp(-My * self.time_intervals * (z - mx) / (Mx - mx))
            y_ = maximum(y_, self.min_p_array)
            probs[s] = (x_, y_, pdf)
        return probs

    def personal_work_hours(self, uid=0):
        """
        Finds the probability distribution of working hours of a user

        :param uid: (defaul=0) the user id
        :return: the array of probabilities, indexed by 24 hours
        """
        from numpy import array
        hours = [0. for _ in range(24)]
        joe_df = self.same_df[self.same_df['user_id'] == uid]
        for hr in range(24):
            work = joe_df[(joe_df['numeric_time'] >= hr) & (joe_df['numeric_time'] < hr + 1)]
            hours[hr] = len(work)
        return array(hours) / sum(hours)

    def group_work_hours(self, uid=[0]):
        """
        Finds the probability distribution of working hours of a group of users

        :param uid: (defaul=[0]) a list of  user ids
        :return: the array of probabilities, indexed by 24 hours
        """
        from numpy import array
        hours = [0. for _ in range(24)]
        for usr in uid:
            joe_df = self.same_df[self.same_df['user_id'] == usr]
            for hr in range(24):
                work = joe_df[(joe_df['numeric_time'] >= hr) & (joe_df['numeric_time'] < hr + 1)]
                hours[hr] += len(work)
        return array(hours) / sum(hours)


################################################################
class NaiveBayes(object):
    """
    A customized implementation of the Naive Bayes classifier.

    :param ratio: (default=0.146) the factor compensating for imbalance in the data
    :param joe_id: (default=0) the id of the user to be identified
    :param ids: (default=None) the id of similar users
    :param min_visits: (default=10) the minimum number of times that a website needs to be visited to be considered as
        frequently visited
    :param time_intervals: (default=50) number of sub-intervals for time spent on each site
    :param min_p: minimum value of the probability replacing 0.
    :param df: the pandas DataFrame to be used for feature extraction. If df is None, then the contents
        of `dataset.json` will be used.
    """

    def __init__(self,
                 ratio=.146,
                 joe_id=0,
                 ids=None,
                 min_visits=10,
                 time_intervals=50,
                 min_p=1e-5,
                 df=None):
        self.ratio = ratio
        self.joe_id = joe_id
        if ids is None:
            self.ids = [0, 15, 56, 69, 82, 105, 111, 181, 192]
        else:
            self.ids = ids
        self.min_visits = min_visits
        self.time_intervals = time_intervals
        self.min_p = min_p
        self.site_info = {}
        for uid in self.ids:
            self.site_info[uid] = SiteInfo(joe_id=uid,
                                           min_visits=self.min_visits,
                                           time_intervals=self.time_intervals,
                                           min_p=self.min_p, df=df)
        self.p_sites_joe = None
        self.hours_joe = None
        self.p_sites = {}
        self.hours = {}
        self.p_sites_other = None
        self.hours_other = None
        self.p_joe = None
        self.p_users = {}
        self.p_other = None
        self.confusion = {}

    def fit(self):
        """
        Finds all distributions and other necessary features.

        :return: self
        """
        total_n = len(self.site_info[self.ids[0]].same_df)
        for uid in self.ids:
            uid_n = len(self.site_info[uid].joe_df)
            self.p_users[uid] = float(uid_n) / total_n
        joe_n = len(self.site_info[0].joe_df)
        self.p_joe = float(joe_n) / total_n
        self.p_other = 1. - self.p_joe
        for uid in self.ids:
            self.p_sites[uid] = self.site_info[uid].site_usage_prob(user_id=uid)
            self.hours[uid] = self.site_info[uid].personal_work_hours(uid=uid)
        self.p_sites_joe = self.site_info[0].site_usage_prob(user_id=self.joe_id)
        self.hours_joe = self.site_info[0].personal_work_hours(uid=0)
        self.p_sites_other = self.site_info[0].site_usage_prob(user_id=None, exclude=self.joe_id)
        self.hours_other = self.site_info[0].group_work_hours(uid=self.ids[1:])
        return self

    def _estimate_prob(self, dist, t):
        """
        Finds the probability associated to `t` based on `dist`

        :param dist: the visit time distribution
        :param t: the visit time
        :return: the corresponding probability
        """
        x = dist[0]
        p = dist[1]
        pdf = dist[2]
        # return pdf(t)
        if (t < x[0]) or (t > x[-1]):
            return self.min_p
        idx = 0
        while idx < self.time_intervals:
            if t <= x[idx + 1]:
                return p[idx]
            idx += 1

    def predict(self, row):
        """
        predicts the classification of a session

        :param row: a row (session record) from a acceptable DataFrame like 'dataset.json' or 'verify.json'
        :return: (pure Naive Bayes prediction, ratio of probabilities, balanced Naive Base, user_id classification)
        """
        session = row['sites']
        hour = int(row['numeric_time'])
        # process session
        page_length = {}
        for rec in session:
            page_length[rec['site']] = rec['length']
        # for Joe
        p_joe = self.p_joe * self.hours_joe[hour]
        for page in self.p_sites_joe:
            if page in page_length:
                p_joe *= self._estimate_prob(self.p_sites_joe[page], page_length[page])
            else:
                p_joe *= self.min_p
        # for others:
        p_others = self.p_other * self.hours_other[hour]
        for page in self.p_sites_other:
            if page in page_length:
                p_others *= self._estimate_prob(self.p_sites_other[page], page_length[page])
            else:
                p_others *= self.min_p
        # for each user
        sel_uid = self.ids[0]
        sel_up = 0.
        ps = {}
        for uid in self.ids:
            p_uid = self.p_users[uid] * self.hours[uid][hour]
            for page in self.p_sites[uid]:
                if page in page_length:
                    p_uid *= self._estimate_prob(self.p_sites[uid][page], page_length[page])
            if p_uid > sel_up:
                sel_up = p_uid
                sel_uid = uid
            ps[uid] = p_uid
        r = p_joe / p_others
        if r >= self.ratio:
            pr = 0
        else:
            pr = 1
        if p_joe >= p_others:
            return 0, r, pr, sel_uid
        else:
            return 1, r, pr, sel_uid

    def score(self, df, measure='accuracy'):
        """
        Finds a performance measure based on a given DataFrame

        :param df: a labeled DataFrame
        :param measure: (default='accuracy) the performance measure: 'accuracy', 'f1', 'mix' that is the average
            of accuracy and f1
        :return: the value of measure
        """
        TP, TN, FP, FN = 0, 0, 0, 0
        for idx, row in df.iterrows():
            uid = row['user_id']
            p, r, pr, upr = self.predict(row)
            if (pr == 0) & (uid == 0):
                TP += 1
            elif (pr == 0) & (uid != 0):
                FP += 1
            elif (pr != 0) & (uid == 0):
                FN += 1
            elif (pr != 0) & (uid != 0):
                TN += 1
        self.confusion = dict(TP=TP, TN=TN,
                              FP=FP, FN=FN)
        if measure == 'accuracy':
            return (TP + TN) / (TP + TN + FP + FN)
        elif measure == 'f1':
            return 2 * TP / (2 * TP + FP + FN)
        elif measure == 'mix':
            return ((TP + TN) / (TP + TN + FP + FN) + 2 * TP / (2 * TP + FP + FN)) / 2


################################################################
# Timezone adjustment
timezone_exchange = {'Spain/Madrid': 'Europe/Madrid', 'France/Paris': 'Europe/Paris',
                     'Australia/Sydney': 'Australia/Sydney',
                     'Netherlands/Amsterdam': 'Europe/Amsterdam', 'China/Shanghai': 'Asia/Shanghai',
                     'Japan/Tokyo': 'Asia/Tokyo',
                     'Russia/Moscow': 'Europe/Moscow', 'India/Delhi': 'Asia/Kolkata',
                     'Singapore/Singapore': 'Asia/Singapore',
                     'UK/London': 'Europe/London', 'USA/Miami': 'US/Eastern',
                     'Malaysia/Kuala Lumpur': 'Asia/Kuala_Lumpur',
                     'USA/San Francisco': 'US/Pacific', 'USA/Chicago': 'US/Michigan',
                     'Canada/Toronto': 'Canada/Eastern',
                     'USA/New York': 'US/Eastern', 'Germany/Berlin': 'Europe/Berlin',
                     'New Zealand/Auckland': 'Pacific/Auckland',
                     'Italy/Rome': 'Europe/Rome', 'Brazil/Rio de Janeiro': 'Brazil/East',
                     'Canada/Vancouver': 'Canada/Pacific'}


def utc_to_tz(dt, tzn):
    """
    convert GMT to local time

    :param dt: a date-time in GMT
    :param tzn: time zone
    :return: conversion of 'dt' from GMT into 'tzn' timezone
    """
    from dateutil import tz
    cdt = dt.replace(tzinfo=tz.tzutc()).astimezone(tz.gettz(timezone_exchange[tzn]))
    return cdt


class CatchJoe(object):
    """
    The main class that filters records and classify them

    :param joe_id: (default=0) the user whose records to be identified
    :param insight: (default=True) whether to use Oracle's help or not
    """

    def __init__(self, joe_id=0, insight=True, output='results.txt'):
        from pandas import read_json
        from datetime import datetime
        self.joe_id = joe_id
        self.insight = insight
        self.output = output
        if self.joe_id != 0:
            self.insight = False
        self.Joes_last_date = datetime(year=2017, month=9, day=4)
        self.Joes_first_date_2_move = datetime(year=2017, month=5, day=16)
        self.third_location = 'Canada/Toronto'
        self.Joes_first_date_3_move = datetime(year=2017, month=12, day=8)
        self.forth_location = 'Singapore/Singapore'
        self.Joes_first_date_4_move = datetime(year=2018, month=10, day=1)
        self.Joes_average_length_stay = 250
        self.train_df = self.process_data(read_json('dataset.json'))
        self.test_df = self.process_data(read_json('verify.json'))

    def process_data(self, df):
        """
        process, add and make features such as local date-time for the DataFrame

        :param df: The DataFrame to be processed
        :return: the processed DataFrame
        """
        from datetime import datetime
        from time import strptime
        df['DateTime'] = df.apply(lambda x: datetime.strptime("%d-%d-%d %s" % (x['date'].year,
                                                                               x['date'].month,
                                                                               x['date'].day,
                                                                               x['time']),
                                                              "%Y-%m-%d %H:%M:%S"),
                                  axis=1)
        df['AdjDateTime'] = df.apply(lambda x: utc_to_tz(datetime.strptime("%d-%d-%d %s" % (x['date'].year,
                                                                                            x['date'].month,
                                                                                            x['date'].day,
                                                                                            x['time']),
                                                                           "%Y-%m-%d %H:%M:%S"), x['location']),
                                     axis=1)
        df['numeric_time'] = df.apply(lambda x: strptime(str(x['AdjDateTime'].time()),
                                                         "%H:%M:%S")[3] + strptime(str(x['AdjDateTime'].time()),
                                                                                   "%H:%M:%S")[4] / 100.,
                                      axis=1)
        df.sort_values(by=['DateTime'], inplace=True)
        return df

    def init_classifier(self, ratio=.14):
        """
        Initialize the Naive Bayes classifier

        :param ratio: (default=0.14) the imbalance factor
        :return: None
        """
        self.nb = NaiveBayes(ratio=ratio, df=self.train_df)
        self.nb.fit()

    def crude_filter(self, row):
        """
        A basic filter to sieve none-similar records

        :param row: record to be sieved
        :return: 1 if discarded (not similar) and 0 if similar
        """
        if row['gender'] != 'm':
            return 1
        if row['locale'] != 'ru_RU':
            return 1
        if row['os'] not in ['Ubuntu', 'Windows 10']:
            return 1
        if row['browser'] not in ['Chrome', 'Firefox']:
            return 1
        if row['numeric_time'] <= 10:
            return 1
        if (row['numeric_time'] >= 15) and (row['numeric_time'] <= 19):
            return 1
        return 0

    def location_insight(self, row):
        """
        Hints about location of Joe

        :param row: a session record
        :return: 1 if not Joe, 0 if Joe and -1 if not certain
        """
        if ((self.Joes_average_length_stay - (row['date'] - self.Joes_first_date_2_move).days) > 0) and (
                row['location'] == self.third_location):
            return 0
        elif ((self.Joes_average_length_stay - (row['date'] - self.Joes_first_date_2_move).days) > 0) and (
                row['location'] != self.third_location):
            return 1
        elif ((row['date'] >= self.Joes_first_date_3_move) and (row['date'] <= self.Joes_first_date_4_move) and (
                row['location'] == self.forth_location)):
            return 0
        elif ((row['date'] >= self.Joes_first_date_3_move) and (row['date'] <= self.Joes_first_date_4_move) and (
                row['location'] != self.forth_location)):
            return 1
        else:
            return -1

    def predict(self):
        """
        Adds a column to the test data with classification labels

        :return: test data with classification of each session
        """
        from pandas import DataFrame
        clmns = list(self.test_df.columns)
        pred_dict = {clm: [] for clm in clmns}
        pred_dict['Joe'] = []
        try:
            fl = open(self.output, 'w')
        except:
            fl = None
        for idx, row in self.test_df.iterrows():
            if self.crude_filter(row) == 0:
                if self.insight:
                    loc_ins = self.location_insight(row)
                    if loc_ins == 0:
                        row['Joe'] = 0
                    elif loc_ins == 1:
                        row['Joe'] = 1
                    else:
                        rslt = self.nb.predict(row)
                        row['Joe'] = rslt[2]
                else:
                    rslt = self.nb.predict(row)
                    row['Joe'] = rslt[2]
            else:
                row['Joe'] = 1
            for clm in clmns:
                pred_dict[clm].append(row[clm])
            pred_dict['Joe'].append(row['Joe'])
            if fl is not None:
                fl.write("%d\n" % row['Joe'])
        pred_df = DataFrame(pred_dict)
        return pred_df


wizard = CatchJoe(insight=True)
wizard.init_classifier(ratio=.14)
results_df = wizard.predict()
orig_cols = ['browser', 'date', 'gender', 'locale', 'location', 'os', 'sites', 'time', 'Joe']
results_df[orig_cols].to_csv('pred.csv', index=False)