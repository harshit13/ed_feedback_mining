'''
    author Rishabh Goel and Aakash Deep
    CSE544 Spring 2020 Project
Motivation: FINDING SIMILAR SCHOOLS OR AREAS TO SUGGEST INSIGHTS
    FOR SDG 4
Task: Find similar schools from survey feedback collected from students
Approach:
    - collect feed_backs for all available years, stored on HDFS
    - Extract 
    - REMOVE not required columns
    - FILL null values with most popular or mode of that variable in that school
    - Generate characteristic matrix from feedbacks by creating 
        buckets[0-20%, 21-40%, 41-60%, 61-80%, 81-100%] for selected answer in that school.
    - Create signature matrix from chracterstic matrix.
    - Find similar area/schools using LSH.
Data Pipeline and algo used
    - PySpark -  to use rdd and apply transformation
    - HDFS - replication factor=3 [For Spark Processing on GCP Data Proc]
    - Similarity search using LSH

System Used:
    GCP cluster with 1 master node 2 worker node
    Image Version:  1.4 (Debian 9, Hadoop 2.9, Spark 2.4)

    Master node             Standard (1 master, N workers)
    Machine type            e2-standard-2
    Primary disk type     pd-standard
    Primary disk size      64GB
    Worker nodes            2
    Machine type            e2-highmem-4
    Primary disk type       pd-standard
    Primary disk size       32GB
'''


# Necessary imports
import numpy as np
from scipy import stats
import pprint as pp
from pyspark import SparkContext, SparkConf
import math

conf = SparkConf().setAppName("myFirstApp").setMaster("local")
sc = SparkContext(conf=conf)

# read sample data csv 
rdd = sc.textFile('hdfs:/sampleDataSet.csv')

# # Data Cleaning

# extract features from the first row of csv 
features = sc.broadcast(rdd.take(1)[0].split(","))

# converting each row from string to a list
df = rdd.map(lambda x : x.split(","))

# dropping header row from csv data
df = df.filter(lambda x : x[1]!='"CNT"')

# keeping 7th feature (Area Code) or 3rd feature (School ID) and 15th-321st features (Feedback variables)
# Area code will be used as columns and feedback variables for shingles
criteria_feat_idx = 6 #or 2 for school ID

feat = [features[criteria_feat_idx]]+features[14:320]
df = df.map(lambda x : (x[criteria_feat_idx], x[14:320]))

# grouping data i.e. feedback of students by area code 
# input  : (area_code,feedback11,feedback12,.....,feedback1n)
# output : (area_code,[[feedback11,feedback12,...],[feedback21,feedback22,...],...[feedbackm1,feedbackm2,...]])
df_sch = df.groupByKey().map(lambda x : (x[0], list(x[1])))

# reordering feedback data to form lists of each type of feedback 
# input  : (area_code,[[feedback11,feedback12,...],[feedback21,feedback22,...],...[feedbackm1,feedbackm2,...]])
# output : (area_code,[[feedback11,feedback21,...],[feedback12,feedback22,...],...[feedback1m,feedback2m,...]])
df_sch_feat = df_sch.map(lambda x : (x[0],np.transpose(x[1])))

# filling missing values with the most popular feedback of that type
df_sch_feat_mode = df_sch_feat.map(lambda x: (x[0],[ [stats.mode(arr)[0][0] if (x=='NA' and stats.mode(arr)!='NA') else x for x in arr]  for arr in x[1]] ))


# # MinHashing


# finding the percentage of students who gave that feedback in its category(question asked)
# input  : (area_code,[[feedback11,feedback21,...],[feedback12,feedback22,...],...[feedback1m,feedback2m,...]])
# output : (area_code,[(feedback_ques_index,feedback,percentage_students_per_area_per_feedback_per_question),....])
df_sch_perc = df_sch_feat_mode.map(lambda x : (x[0], [ (i,elem,100*arr.count(elem)/len(arr)) for i,arr in enumerate(x[1]) for elem in set(arr)  ]  ))

# mapping the percentage of each feedback to the nearest (20*i)% range, say 15.6->20% , 32->40% , 96->100%
df_sch_cat = df_sch_perc.map(lambda x : (x[0],[(a[0],a[1],math.ceil(a[2]/20)*20 )for a in x[1]]))



'''Instead of using k hash functions, we are dividing the shingles into batches and using min hash values from each batch
each batch consists of 5 questions feedbacks , one shingle is based on (question, feedback, percentage).
Say, we have 200 questions data, then we divide them into batches of 5 questions each, each batch has student's feedback of
respective 5 questions. To capture the popularity of a feedback on a specific question, we represent a shingle by the question id,
feedback option and percentage of students who answered that feedback for that question in that area.'''
def minHash(record):
 l = {}
 for v in record:
   p = int((v[0]-1)/5)
   h = hash(v)
   try:
     if l[p] > h:
       l[p] = h
   except KeyError:
     l[p] = h
 return l
 
# finding the minhash from the hash values calculated from (question_feedback_percentage) as a shingle
# input  : (area_code,[(feedback_ques_index,feedback,percentage_students_per_area_per_feedback_per_question),....])
# output : (area_code,{signature_index : minhash_value})
df_sch_min_hash = df_sch_perc.map(lambda t : (t[0], minHash(t[1])  ))

# converting dictionary of min hash values to list of hash values
# input  : (area_code,{signature_index : minhash_value})
# output : (area_code,[min_hash_values])
df_sch_band = df_sch_min_hash.map(lambda x : (x[0], [v for k,v in x[1].items()]))


# # Locality Sensitive Hashing



# performing lsh first step by hashing bands of the signatures for each column
# We had 61 signatures for each area code, after experimenting with different band sizes, we finally
# decided to divide them into bands of size 5
# input  : (area_code,[min_hash_values])
# output : (area_code,[hash values of bands])
r = sc.broadcast(5)
df_sch_band_group = df_sch_band.map(lambda x : (x[0],[ hash(tuple(x[1][i:i+r.value])) for i in range(0,len(x[1]),r.value) ]))

# finding similar sets of area codes by using signatures from the signature matrix
# input  : (area_code,[hash values of bands])
# output : (signature_index,[(area_code,hash_value),.....])
df_sch_lsh = df_sch_band_group.flatMap(lambda x : [(i,(x[0],x[1][i])) for i in range(len(x[1]))]  ).groupByKey().map(lambda x : (x[0], list(x[1])))

from collections import defaultdict
def findSim(x):
    l = defaultdict(list)
    for areaCode, hashVal in x:
        l[hashVal].append(areaCode)
    ret = []
    for v in l.values():
        if len(v) > 1:
            ret.append(set(v))
    return ret

# input  : (signature_index,[(area_code,hash_value),.....])
# output : [{similar area code candidates},...]
df_sch_band_similar = df_sch_lsh.flatMap(lambda x: findSim(x[1]))

sim_candidates = df_sch_band_similar.collect()


# pp.pprint(sim_candidates)


# finding candidate pairs from sets of similar area codes
candidatePairs = set()

for s in sim_candidates:
    p = list(s)
    for pi in p:
        for pj in p:
            if pi != pj:
                candidatePairs.add((min(pi,pj),max(pi,pj)))
                    
                    
pp.pprint(candidatePairs)


# Out of 1400*1400 .i.e 1960000 possible pairs, we got 228494 candidate pairs, i.e around 33% of the total possible pairs
# print(len(candidatePairs))




with open("result11.txt","w") as f:
    for l in candidatePairs:
        f.write(str(l))



# following code is to calculate jaccard similarity using cartesian product. We just wanted
# to see run time difference and result difference between LSH and fundamental jaccard similarity. 
# It was executed on small subset of data.

# # # Jaccard Similarity

# df_cmp = df_sch_band.cartesian(df_sch_band)


# df_jaccard = df_cmp.map(lambda x : ((x[0][0],x[1][0]),list(np.array(x[0][1]) == np.array(x[1][1])).count(True)/len(x[0][1]))  )

# df_jaccard1 = df_jaccard.filter(lambda x: x[0][0]<x[0][1])

# df_jaccard_sorted = df_jaccard1.map(lambda x :(x[1],(x[0]))).sortByKey("desc")

# df_jaccard_top = df_jaccard_sorted.take(len(candidatePairs))

# df_jaccard_top_pairs = [a[1] for a in df_jaccard_top]

# common_pairs = set(df_jaccard_top_pairs).intersection(candidatePairs)

