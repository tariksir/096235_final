import numpy as np
import pandas as pd
import os
import time
import random
import itertools
from scipy.stats import spearmanr


def normalized_agreement_score(questions_list, qk, s1, s2):
    s1_rank = s1[s1['question_unique'] == qk]['Difficulty'].iloc[0]
    s2_rank = s2[s2['question_unique'] == qk]['Difficulty'].iloc[0]

    Zk = s2[(s2['question_unique'].isin(questions_list)) & (s2['Difficulty'] <= s2_rank)]['question_unique'].unique()
    if len(Zk) == 0:
        return 0
    k = len(Zk)

    A_k = len(s1[(s1['question_unique'].isin(Zk)) & (s1['Difficulty'] <= s1_rank)])
    k = len(s1[(s1['question_unique'].isin(Zk))])
    score = A_k / k
    return score


def AP_score(questions_list, s1, s2):
    score = 0
    s1 = s1[s1['question_unique'].isin(questions_list)]
    s2 = s2[s2['question_unique'].isin(questions_list)]
    for qk in questions_list:
        score += normalized_agreement_score(questions_list, qk, s1, s2)
    return score / len(questions_list)


def AP_score_update(questions_list, s1, s2, prev_score, new_q, weight=0.2):
    s1 = s1[s1['question_unique'].isin(questions_list)]
    s2 = s2[s2['question_unique'].isin(questions_list)]
    new_q_ak = normalized_agreement_score(questions_list, new_q, s1, s2)
    new_score = ((1 - weight) * prev_score) + (weight * new_q_ak)
    return new_score


#############
def gold_standard_score(answer_stats):
    if answer_stats['Correct First Attempt'] == 1: return 1
    return 1 - 0.2 * answer_stats['Incorrects']


def get_ranked_questions(df):
    df['Difficulty'] = df.apply(lambda row: gold_standard_score(row), axis=1)
    return df


###############

def relative_voting(qk, ql, students_df, sim):
    score = 0
    students = students_df['Anon Student Id'].unique()
    for j in students:
        s_j = students_df[students_df['Anon Student Id'] == j]
        score += sim[j] * np.sign(s_j[s_j['question_unique'] == ql]['Difficulty'].iloc[0] -
                                  s_j[s_j['question_unique'] == qk]['Difficulty'].iloc[0])

        if abs(score) > 3:
            break

    return np.sign(score)


class Edurank():
    def __init__(self, s_id, Q_i, Q, memory_size=5):
        self.s_id = s_id
        self.Q = Q
        self.Q_i = Q_i
        self.similarities = {}
        self.memory_size = memory_size
        self.similarities_top = {}
        self.S_top = pd.DataFrame()
        self.params = dict(memory_size=self.memory_size)

    def rank_q(self):
        ranked = []
        sim = self.similarities_top
        s_top = self.S_top
        rvs = {}

        # compute a relative voting
        for q1 in range(len(self.Q_i)):
            students = sim.keys()
            relevant_students_q = s_top[(s_top['Anon Student Id'].isin(students)) &
                                        (s_top['Problem Name'] == self.Q_i[q1])]['Anon Student Id'].unique()
            Li_no_q = \
                s_top[(s_top['Problem Name'] != self.Q_i[q1]) & (s_top['Anon Student Id'].isin(relevant_students_q))][
                    'Problem Name'].unique()
            rvs[q1] = {}
            print(q1)
            for q2 in range(q1, len(self.Q_i)):
                if self.Q_i[q2] not in Li_no_q:
                    continue
                relevant_students_ql = s_top[(s_top['Anon Student Id'].isin(relevant_students_q)) &
                                             (s_top['Problem Name'] == self.Q_i[q2])]['Anon Student Id'].unique()

                if len(relevant_students_ql) == 0:
                    continue
                relevant_S = s_top[s_top['Anon Student Id'].isin(relevant_students_ql)]
                rvs[q1][q2] = relative_voting(self.Q_i[q1], self.Q_i[q2], relevant_S, sim)

                if q2 not in rvs:
                    rvs[q2] = {}
                rvs[q2][q1] = rvs[q1][q2]

        # compute a copeland score based on the rv
        for q1 in range(len(self.Q_i)):
            sum_copeland = 0
            if q1 in rvs:
                for q2 in range(q1, len(self.Q_i)):
                    if q2 in rvs[q1]:
                        sum_copeland += rvs[q1][q2]

            q_ranked = {}
            q_ranked['Anon Student Id'] = self.s_id
            q_ranked['question_unique'] = self.Q_i[q1]
            q_ranked['Difficulty'] = sum_copeland
            ranked.append(q_ranked)

        return pd.DataFrame(ranked)

    def update_model(self, new_Q):
        self.S_top = self.S_top.append(new_Q, ignore_index=True)
        s_i = self.S_top[self.S_top['Anon Student Id'] == self.s_id]
        self.Q = [x for x in self.Q if x not in new_Q['question_unique'].unique()]

        # iterate student that ansered new_q
        for j in self.similarities_top.keys():
            s_j = self.S_top[self.S_top['Anon Student Id'] == j]
            # join all i and j quesions
            L = pd.merge(s_j, s_i, how='inner', on=['question_unique'])['question_unique'].unique()
            for new_q in new_Q['question_unique'].unique():
                if new_q not in L:
                    continue
                prev_score = self.similarities_top[j]
                # updating only 1 question
                self.similarities_top[j] = AP_score_update(L, s_i, s_j, prev_score, new_q)
                print("sim changes:")
                print(abs(prev_score - self.similarities_top[j]))

    def fit_model(self, S):
        self.similarities = {}

        s_i = S[S['Anon Student Id'] == self.s_id]
        # iterate student j that have questions for questionaire and from previous questions
        for student_j in S[(S['Anon Student Id'] != self.s_id) & (S['question_unique'].isin(self.Q))][
            'Anon Student Id'].unique():
            self.similarities[student_j] = 0
            counter = 0
            s_j = S[S['Anon Student Id'] == student_j]
            # check other questions join
            join_questionnaires = pd.merge(s_j, s_i, how='inner', on=['unit'])['unit'].unique()
            for qusare in join_questionnaires:
                s_j_q = s_j[s_j['unit'] == qusare]
                s_i_q = s_i[s_i['unit'] == qusare]
                L = pd.merge(s_j_q, s_i_q, how='inner', on=['question_unique'])['question_unique'].unique()
                if len(L) > 1:
                    self.similarities[student_j] += AP_score(L, s_i_q, s_j_q)
                    counter += 1
            if counter == 0:
                del self.similarities[student_j]
            else:
                self.similarities[student_j] = self.similarities[student_j] / counter

        if not self.similarities:
            raise Exception('studnt dont share items with others')
        self.similarities = {key: value for key, value in self.similarities.items() if value != {}}

        self.S_top = S[S['question_unique'].isin(self.Q_i)]
        similiarities_relevant = {your_key: self.similarities[your_key] for your_key in
                                  self.S_top[self.S_top['Anon Student Id'].isin(self.similarities.keys())][
                                      'Anon Student Id'].unique()}
        print(similiarities_relevant)
        self.similarities_top = dict(
            sorted(similiarities_relevant.items(), key=lambda item: item[1], reverse=True)[:self.memory_size])


def get_questionnaires(ty, S, questionnaires_to_eval=4):
    if ty == 'fixed':
        return ['Unit CTA1_10', 'Unit ES_04', 'Unit CTA1_10', 'Unit ES_07']
    questionnaires = S['unit'].unique()
    random.seed = 10
    random_questionnaires = [random.randint(0, len(questionnaires) - 1) for r in range(questionnaires_to_eval)]
    random_questionnaires = [questionnaires[s] for s in random_questionnaires]
    return random_questionnaires


def get_data(amount=0):
    answers_path_pp = 'data/algebra_2005_2006_train_pp' + str(amount) + '.csv'
    if os.path.isfile(answers_path_pp):
        S = pd.read_csv(answers_path_pp)
    else:
        if amount == 0:
            S = pd.read_table('data/algebra_2005_2006_train.txt')[
                ['Anon Student Id', 'Problem Hierarchy', 'Problem Name', 'Step Name', 'Correct First Attempt',
                 'Incorrects']]
        else:
            S = pd.read_table('data/algebra_2005_2006_train.txt')[
                    ['Anon Student Id', 'Problem Hierarchy', 'Problem Name', 'Step Name', 'Correct First Attempt',
                     'Incorrects']][:amount]
        problem_df = S['Problem Hierarchy'].apply(lambda x: pd.Series([i for i in reversed(x.split(','))]))
        problem_df.rename(columns={1: 'unit', 0: 'quest'}, inplace=True)
        S = pd.concat((problem_df, S), axis=1)
        S['question_unique'] = S['unit'] + S['Problem Name'] + S['Step Name']
        S = get_ranked_questions(S)
        S.to_csv(answers_path_pp)
    return S


def get_index_product(params):
    i = 0
    params_index = {}
    for k, v in params.items():
        params_index[k] = i
        i += 1
    params_list = [None] * len(params_index.values())
    for name, loc in params_index.items():
        params_list[loc] = params[name]

    params_product = list(itertools.product(*params_list))
    params_product_dicts = []
    for params_value in params_product:
        params_dict = {}
        for param_name, param_index in params_index.items():
            params_dict[param_name] = params_value[param_index]
        params_product_dicts.append(params_dict)

    return params_product_dicts


def get_rate_results(Q_i, standard_rate, rank_result1, rank_result2=None):
    rank_result = {}
    rank_result['total_questions'] = len(Q_i)
    rank_result['SAP_offline'] = AP_score(Q_i, standard_rate, rank_result1)
    rank_result['spearman_offline'], p = spearmanr(standard_rate['Difficulty'], rank_result1['Difficulty'],
                                                   nan_policy='omit')

    if rank_result2 is not None:
        rank_result['SAP_online'] = AP_score(Q_i, standard_rate, rank_result2)
        rank_result['SAP_improve'] = rank_result['SAP_online'] - rank_result['SAP_offline']
        rank_result['spearman_online'], p = spearmanr(standard_rate['Difficulty'], rank_result2['Difficulty'],
                                                      nan_policy='omit')
        rank_result['spearman_improve'] = rank_result['spearman_online'] - rank_result['spearman_offline']

    return rank_result


def evaluate(sequencer_class, model_params, amount, results_path, amount_q=5, amount_s=3):
    S = get_data(amount)
    questionnaires = get_questionnaires('fixed', S)[:amount_q]

    all_results = []
    for questionnaire in questionnaires:
        print("start evaluate questionaire: " + questionnaire)
        Q = list(S.loc[S['unit'] == questionnaire]['question_unique'].unique())
        students_i = S[S['unit'] == questionnaire]['Anon Student Id'].unique()[:amount_s]
        for student_i in students_i:
            Q_i = list(
                S[(S['Anon Student Id'] == student_i) & (S['question_unique'].isin(Q))]['question_unique'].unique())
            S_simulation = S[(S['Anon Student Id'] != student_i) | (
                    (S['Anon Student Id'] == student_i) & (~S['question_unique'].isin(Q)))]

            model_params['s_id'] = [student_i]
            model_params['Q_i'] = [Q_i]
            model_params['Q'] = [Q]

            models = []
            model_params_product = get_index_product(model_params)
            for model_param in model_params_product:
                models.append(sequencer_class(**model_param))

            for question_sequencer in models:
                print("evaluating " + sequencer_class.__name__ + str(question_sequencer.params))
                start_time = time.time()

                question_sequencer.fit_model(S_simulation)
                fit_dur = round((time.time() - start_time) / 60.0, 3)

                model_rate = question_sequencer.rank_q(). \
                    sort_values('Difficulty', ascending=True). \
                    drop_duplicates('question_unique', 'first'). \
                    sort_values('question_unique', ascending=True)
                rate_dur = round((time.time() - fit_dur) / 60.0, 3)

                standard_rate = S[(S['Anon Student Id'] == student_i) & (S['question_unique'].isin(Q_i))]. \
                    sort_values('Difficulty', ascending=True). \
                    drop_duplicates('question_unique', 'first'). \
                    sort_values('question_unique', ascending=True)

                rank_result = {}
                rank_result['student_i'] = student_i
                rank_result['fit_duration'] = fit_dur
                rank_result['rate_duration'] = rate_dur
                rank_result['model_class'] = sequencer_class.__name__
                rank_result.update(question_sequencer.params)
                rank_result.update(get_rate_results(Q_i, standard_rate, model_rate))
                all_results.append(rank_result)
                print(rank_result)
                del question_sequencer
    pd.DataFrame(all_results).to_csv(results_path)


def run():
    model_params = {}
    evaluate(Edurank, model_params, 250000, 'data/Edurank_Evaluations.csv')


run()
