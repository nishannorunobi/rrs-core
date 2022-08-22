import utils as utl
import logger as log
#######
#Imports
#All Imports
import nltk
#nltk.download('wordnet')
#need to enable above line when first time running
from nltk.corpus import stopwords
import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#######
## read resumes
resume_job_doc_index_map = {}
df_resume = pd.read_csv('dataset/resumes_mid.csv',sep=',')
total_resume = df_resume.shape[0]
resume_doc_name_list = [f'doc_{x}' for x in range(1,total_resume+1)]
resume_job_doc_index_map['resumes'] = resume_doc_name_list
df_resume['doc_name'] = resume_doc_name_list
df_resume_text = df_resume[['doc_name', 'text']].copy()
df_resume_text = df_resume_text.set_index("doc_name")


resume_dict_text = df_resume_text.T.to_dict()
#log.error(resume_dict_text)
#exit()
## read job circular
df_jobs = pd.read_csv('dataset/job_circulars.csv',sep=',')
total_jobs = df_jobs.shape[0]
job_doc_name_list = [f'doc_{x}' for x in range(total_resume+1,total_resume+total_jobs+1)]
resume_job_doc_index_map['jobs'] = job_doc_name_list
# log.error(resume_job_doc_index_map)
# log.error(job_doc_name_list)
df_jobs['doc_name'] = job_doc_name_list
df_jobs_text = df_jobs[['doc_name', 'text']].copy()
df_jobs_text = df_jobs_text.set_index("doc_name")
jobs_dict_text = df_jobs_text.T.to_dict()
# log.error(jobs_dict_text)
# exit()
#######
#merge two dictionary
resume_dict_text.update(jobs_dict_text)
# log.error(resume_dict_text)

#######
document_freq_dict = utl.get_document_frequency_dictionary(resume_dict_text)
# log.error('#############document_freq_dict')
# log.error(document_freq_dict)
vocabulary_list = utl.get_vocabulary_list(document_freq_dict)

total_documents = total_resume + total_jobs
#log.error(f"############################ total_documents :: {total_documents}")
total_words = len(vocabulary_list)
#log.error(f"############################ total_words :: {total_words}")

#######
# Build Normal Bag of words
bow_df = pd.DataFrame(data=np.array([np.zeros(total_words)]*total_documents).T, 
                          index=vocabulary_list, 
                          columns=document_freq_dict.keys())



for key in document_freq_dict.keys():
  words,counts = document_freq_dict.get(key)
  bow_df.loc[words,key] = counts
# log.error('#############bow_df')
# log.error(bow_df[0:40])

#######
#Build Binary Bag of words, it just ensure if a certain word exists in a document or not
binary_bow_df = pd.DataFrame(data=np.array([np.zeros(total_words)]*total_documents).T, 
                          index=vocabulary_list, 
                          columns=document_freq_dict.keys())

for key in document_freq_dict:
  words,counts = document_freq_dict.get(key)
  binary_bow_df.loc[words,key] = np.array(np.ones(len(counts)))

# log.error('#############binary_bow_df')
# log.error(binary_bow_df)

#######
#Calculate TF matrix
tf_matrix_df = pd.DataFrame(data=np.array([np.zeros(total_words)]*total_documents).T, 
                          index=vocabulary_list, 
                          columns=document_freq_dict.keys())

for key in document_freq_dict:
  words,counts = document_freq_dict.get(key)
  tf_values = counts / counts.sum()
  tf_matrix_df.loc[words,key] = tf_values
#/print tf_matrix

#######
#Calculate IDF matrix
idf_matrix_df = pd.DataFrame(data=binary_bow_df.sum(axis=1), 
                          index=vocabulary_list, 
                          columns=['word_in_docs'])
#log.error(f" document frequency df : occurence of term in total(N) documents :: {idf_matrix_df}")
idf_matrix_df['pre-idf'] = total_documents/idf_matrix_df['word_in_docs']
idf_matrix_df['idf'] = np.log(idf_matrix_df['pre-idf'])
#/print idf_matrix

#######
#Calculate tf-idf matrix
tf_idf = tf_matrix_df[document_freq_dict.keys()].multiply(idf_matrix_df["idf"], axis="index")
#/print tf_idf[2000:2050]
log.error('#############tf_idf')
log.error(tf_idf)
#print(tf_idf[2000:2050])

#######
#Calculate cosine similarity
import operator
job_class_name_list = []
similarity_score_df = pd.DataFrame(index=resume_doc_name_list, columns=job_doc_name_list)
for job_doc in job_doc_name_list:
  job_class_level = f'{job_doc}_class'
  job_class_name_list.append(job_class_level)
  for resume_doc in resume_doc_name_list:
    similarity_score = utl.cosine_similarity(tf_matrix_df[job_doc].tolist(),tf_matrix_df[resume_doc].tolist())
    similarity_score_df.loc[resume_doc,job_doc] = similarity_score
    if job_class_level not in similarity_score_df:
      similarity_score_df[job_class_level] = utl.get_category_level(similarity_score)
    else:
      similarity_score_df.loc[resume_doc,job_class_level] = utl.get_category_level(similarity_score)

log.error('##############similarity_score_df')
log.error(similarity_score_df)
# exit()

#######
# combine topsis criteria matrix for job 1
# log.error(df_resume)

topsis_criteria_df = df_resume[['doc_name', 'university_ranking','years_of_experience']].copy()
topsis_criteria_df = df_resume[['doc_name', 'university_ranking','years_of_experience']].copy()
# log.error(topsis_criteria_df)
# exit()
"""
topsis_criteria_df = pd.DataFrame(
    df_resume['doc_name'].values
    ,columns=['doc_name'])

topsis_criteria_df['university_ranking'] = pd.DataFrame(
    df_resume['university_ranking'].values
    ,columns=['university_ranking'])

topsis_criteria_df['years_of_experience'] = pd.DataFrame(
    df_resume['years_of_experience'].values
    ,columns=['years_of_experience'])
"""
x=1
for job_class_level in job_class_name_list:
  topsis_criteria_df['similarity_score_'+str(x)] = pd.DataFrame(
    similarity_score_df[job_class_level].values
    ,columns=[job_class_level])
  x=x+1

"""
topsis_criteria_df['similarity_score_1'] = pd.DataFrame(
    similarity_score_df['job-1'].values
    ,columns=['job-1'])
"""
log.error('###################topsis_criteria_df')
log.error(topsis_criteria_df)
# exit()
#######

#build normalization matrix
sos_univ_rank = np.sqrt((topsis_criteria_df['university_ranking']**2).sum())
sos_experiences = np.sqrt((topsis_criteria_df['years_of_experience']**2).sum())
sos_matching_socre = np.sqrt((topsis_criteria_df['similarity_score_1']**2).sum())

topsis_criteria_df['university_ranking'] /= sos_univ_rank
topsis_criteria_df['years_of_experience'] /= sos_experiences
topsis_criteria_df['similarity_score_1'] /= sos_matching_socre
#print(topsis_criteria_df)

#########
#Determine weight and multiply with decision matrix with each column(criteria)
#here let some importance of the given criteria
weight_univ_rank = .45
weight_experience = .35
weight_matching_score = .20
# total must be 1
topsis_criteria_df['university_ranking'] *= weight_univ_rank
topsis_criteria_df['years_of_experience'] *= weight_experience
topsis_criteria_df['similarity_score_1'] *= weight_matching_score
#print(topsis_criteria_df)

##############

#Pick ideal best and ideal worst value from criteria(column)
best_univ_rank = topsis_criteria_df['university_ranking'].min()
worst_univ_rank = topsis_criteria_df['university_ranking'].max()

best_experience = topsis_criteria_df['years_of_experience'].max()
worst_experience = topsis_criteria_df['years_of_experience'].min()

best_score = topsis_criteria_df['similarity_score_1'].max()
worst_score = topsis_criteria_df['similarity_score_1'].min()

topsis_criteria_df['sos_ideal_best'] = ((topsis_criteria_df['university_ranking'] - best_univ_rank)**2
+ (topsis_criteria_df['years_of_experience'] - best_experience)**2
+ (topsis_criteria_df['similarity_score_1'] - best_score)**2)**.5

topsis_criteria_df['sos_ideal_worst'] = ((topsis_criteria_df['university_ranking'] - worst_univ_rank)**2
+ (topsis_criteria_df['years_of_experience'] - worst_experience)**2
+ (topsis_criteria_df['similarity_score_1'] - worst_score)**2)**.5

############
#Calculate Performance score & rank
topsis_criteria_df['best_plus_worst'] = topsis_criteria_df['sos_ideal_best'] + topsis_criteria_df['sos_ideal_worst']
topsis_criteria_df['performance'] = topsis_criteria_df['sos_ideal_worst'] / topsis_criteria_df['best_plus_worst']
topsis_criteria_df['rank'] = topsis_criteria_df['performance'].rank(ascending=False)
sorted_topsis_df = topsis_criteria_df.sort_values('rank')
#print(sorted_topsis_df.columns)

###################
#Save overall ranking score
import pickle
over_all_ranking_list = sorted_topsis_df['performance'].to_list()
pickle_out = open("resources/over_all_ranking_list.pickle","wb")
pickle.dump(over_all_ranking_list, pickle_out)
pickle_out.close()

##################
#Save year of experience score
years_of_experience_list = sorted_topsis_df['years_of_experience'].to_list()
pickle_out = open("resources/years_of_experience_list.pickle","wb")
pickle.dump(years_of_experience_list, pickle_out)
pickle_out.close()
#####################
#Save matching score
similarity_socre_list = sorted_topsis_df['similarity_score_1'].to_list()
pickle_out = open("resources/similarity_socre_list.pickle","wb")
pickle.dump(similarity_socre_list, pickle_out)
pickle_out.close()
#####################
#Save resume list
doc_name_list = sorted_topsis_df['doc_name'].to_list()
pickle_out = open("resources/doc_name_list.pickle","wb")
pickle.dump(doc_name_list, pickle_out)
pickle_out.close()
#######################
#dict_df = sorted_topsis_df['doc_name']
#dict_df = sorted_topsis_df['similarity_score_1']
#dict_df = sorted_topsis_df['years_of_experience']
#dict_df = sorted_topsis_df['performance']
#df1 = df[['a', 'b']]
final_df = sorted_topsis_df[['doc_name','performance','similarity_score_1','years_of_experience']]
############################
# convert pandas to resume list save to a file

###########################
#!rm resume_list.pkl
resume_doc_list = final_df.T.to_dict().values()
resume_doc_list = list(resume_doc_list)
#/print resume_doc_list
output_pickle_obj = open('resources/resume_list.pkl', 'wb')
pickle.dump(resume_doc_list, output_pickle_obj)
output_pickle_obj.close()
############################
# convert pandas to resume list save to a file
#!rm resume_dict.pkl

resume_doc = final_df.set_index('doc_name').T.to_dict('list')
output_pickle_obj = open('resources/resume_dict.pkl', 'wb')
pickle.dump(resume_doc, output_pickle_obj)
output_pickle_obj.close()

## calculate category
df_resume_category= df_resume['category'].value_counts().to_dict()
df_resume_category_dict = {}
sum = 0

for key in df_resume_category.keys():
  cat_count = df_resume_category.get(key,0)
  #cat_percent = int(round((cat_count/total_resume)*10)*10);
  cat_percent = int(round((cat_count/total_resume)*100));
  df_resume_category_dict[key] = cat_percent
  sum = sum + cat_percent

## store category
resume_category_list = list(df_resume_category_dict.values())
output_pickle_obj = open('resources/resume_category.pkl', 'wb')
pickle.dump(df_resume_category_dict, output_pickle_obj)
output_pickle_obj.close()
