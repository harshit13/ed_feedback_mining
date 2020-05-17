'''
# @author Shikha
# Here I am applying hypothesis testing to check which feature is more important for a particular SDG goal. 
# I am taking two SDG goals here which are--
# 1) "Proportion of students at the end of lower secondary education achieving at least a minimum proficiency level in reading, both sexes"
# 2) "Proportion of students at the end of lower secondary education achieving at least a minimum proficiency level in mathematics, both sexes"
# Here features are like What is the <highest level of schooling> completed by your father?, How many books are there in your home? etc
# As there are some categorical features as well so have used one hot encoding for data manipulation
# And as Y label, I have scores for each goal for each year for each country, only taking data of year 2015 and 2018
# I took top 20 most co-related and  bottom 20 least co-related features and calculated their P-value
# Platform used to run this code is Google Cloud DataProc running Ubuntu

Concepts used:
	- PySpark : rdd
	- Hypothesis Testing [see from line no 220]


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

import sys
import numpy as np
import math
import pyspark
from scipy import stats

def main(argv):
	sc = pyspark.SparkContext()
	# Two SDG Goals
	goal = ["Proportion of students at the end of lower secondary education achieving at least a minimum proficiency level in reading, both sexes (%)","Proportion of students at the end of lower secondary education achieving at least a minimum proficiency level in mathematics, both sexes (%)"]
	# Two different Y label files for two SDG goals
	files=['hdfs:/data/SDG4_data.csv','hdfs:/data/SDG4_data_Math.csv']

	for i in range(2):

		rdd_SDG_data = sc.textFile(files[i])

		# Here filtered out data for a particular Goal
		def filter_goal(x):
			split_x = x.split(',')
			gol = goal[i]
			if (((split_x[1]+",").replace('"','')+split_x[2].replace('"',''))== gol)\
			and split_x[-4] != "" and split_x[-3] !="" :
				return True
			else:
				return False

		# Here Mapping data for a (country, (year, value))
		def map_cntry_year(x):
			split_x = x.split(',')
			return (split_x[-7].replace('"',''),(split_x[-4].replace('"',''),split_x[-3]))

		# Filtered out country having data of both 2015 and 2018 year
		def map_year_2015_2018(x):
			yr_2015,yr_2018=0.0,0.0
			if x[1][0][0]=="2015":
				yr_2015 = float(x[1][0][1])
			else:
				yr_2018 = float(x[1][0][1])
			if x[1][1][0]=="2015":
				yr_2015 = float(x[1][1][1])
			else:
				yr_2018 = float(x[1][1][1])

			return (x[0],(yr_2015,yr_2018))


		mapped_sdg_data = rdd_SDG_data.filter(lambda x:filter_goal(x)).map(lambda x:map_cntry_year(x))\
								.groupByKey().map(lambda x : (x[0], list(x[1]))).filter(lambda x:len(x[1])==2)\
							.map(lambda x: map_year_2015_2018(x))

		#list down all countries having data points, total 57 country are there
		list_cntry_code = mapped_sdg_data.map(lambda x:x[0]).collect()
		brdcast_cntry = sc.broadcast(list_cntry_code)

		# Feedback variable files for year 2018 of students
		rdd_fb_data_2018 = sc.textFile('hdfs:/data/cy07_msu_stu_qqq.csv')

		# Feedback variable files for year 2015 of students
		rdd_fb_data_2015 = sc.textFile('hdfs:/data/cy6_ms_cmb_stu_qqq.csv')

		# map (k,v) as (country,[list of Feedback variables])
		def map_cnt_STFB(x):
			l = x.split(',')
			# 298 are total features
			return (l[1],(l[14:18]+l[20:41]+l[44:52]+l[55:320]))

		# List down all the Feedback variable list, there are 298 features in total to be used for hypothesis testing
		brdcast_FB_var = sc.broadcast(rdd_fb_data_2018.map(lambda x:map_cnt_STFB(x)).map(lambda x:x[1]).take(1))
		# print(len((brdcast_FB_var.value)[0]))


		# There were multiple entries for each country for each Feedback variable
		# Filled the missing value with the mean and as output, (country,(Feedback_var name,value))
		def take_avg_per_cntry_FB_var(x):
			N = len(x[1])
			dict_count_empty={}
			dict_count_val = {}
			for j in range(N):
				l  = x[1][j]
				for ele in range(len(l)):
					if not l[ele]:
						if ele in dict_count_empty.keys():
							dict_count_empty[ele] +=1.0
						else:
							dict_count_empty[ele] =1.0
					else:
						if ele in dict_count_val.keys():
							dict_count_val[ele] +=float(l[ele])
						else:
							dict_count_val[ele] =float(l[ele])

			FB_var_list = (brdcast_FB_var.value)[0]
			ans=[]
		
			for i in dict_count_val.keys():
				if i not in dict_count_empty.keys():
					ans.append((x[0],(FB_var_list[i],dict_count_val[i]/N)))
				else:
					ans.append((x[0],(FB_var_list[i],(dict_count_val[i]+((dict_count_val[i]/(N-dict_count_empty[i]))*dict_count_empty[i]))/N)))

			return ans

		# Filter Feedback variable data for countries for which we have SDG goal value
		def filter_country(x):
			cntry_list = brdcast_cntry.value
			l = x.split(',')
			if l[1] in cntry_list:
				return True
			else:
				return False


		filter_FB_2018 = rdd_fb_data_2018.filter(lambda x: filter_country(x)).map(lambda x:map_cnt_STFB(x)).groupByKey()\
							.map(lambda x : (x[0], list(x[1])))\
							.flatMap(lambda x:take_avg_per_cntry_FB_var(x))

		filter_FB_2015 = rdd_fb_data_2015.filter(lambda x: filter_country(x)).map(lambda x:map_cnt_STFB(x)).groupByKey()\
							.map(lambda x : (x[0], list(x[1])))\
							.flatMap(lambda x:take_avg_per_cntry_FB_var(x))


		# Map (k,v) pair as (Feedback_varname, (yr, its value))
		def merge_2018(k):
			yr_2018 = k[1][0]
			FB_var = k[1][1]
			return (FB_var[0],(yr_2018[1],FB_var[1]))

		# Map (k,v) pair as (Feedback_varname, (yr, its value))
		def merge_2015(k):
			yr_2015 = k[1][0]
			FB_var = k[1][1]
			return (FB_var[0],(yr_2015[0],FB_var[1]))

		# Merge all SDG goal output with Feedback variable data 
		merge_year_2018_data = mapped_sdg_data.join(filter_FB_2018).map(lambda k:merge_2018(k))
		merge_year_2015_data = mapped_sdg_data.join(filter_FB_2015).map(lambda k:merge_2015(k))

		# Merged all years data with key value as Feedback variable name and in output, (Feedback var, [(goalval, Feedback_var value) for each country])
		total_merge_data = merge_year_2015_data.join(merge_year_2018_data).groupByKey()\
							.map(lambda x : (x[0], list(x[1])))


		# Standandardize data to calculate Beta value for each Feedback variable
		def standard_data_fun(x):
			key = x[0]
			val = x[1]
			ans=[]
			N = len(val)
			arrx=[]
			arry=[]
			for i in range(N):
				arrx.append(val[i][0][1])
				arrx.append(val[i][1][1])
				arry.append(val[i][0][0])
				arry.append(val[i][1][0])
			x_mean = np.mean(arrx)
			y_mean = np.mean(arry)
			x_std = np.std(arrx)
			y_std = np.std(arry)

			ans=[]
			tot = len(arrx)
			beta_val = 0.0
			for i in range(tot):
				beta_val += ((arry[i]-y_mean)/y_std)*((arrx[i]-x_mean)/x_std)
				ans.append(((arry[i]-y_mean)/y_std,(arrx[i]-x_mean)/x_std))
			return ((key,beta_val/(tot-1)),ans)

		def calc_beta_val(x):
			sum=0.0
			N = len(x[1])
			for i in range(len(x[1])):
				sum+=x[1][i][0]*x[1][i][1]

			return (x[0],sum/N-1)



		standardize_data  = total_merge_data.map(lambda x:standard_data_fun(x))

		# Seperated all Feedback variables having 20 most and least co-related variable
		brdcast_top_20_feature = sc.broadcast(standardize_data.sortBy(lambda x: -x[0][1]).map(lambda x:x[0][0]).take(20))
		brdcast_bottom_20_feature = sc.broadcast(standardize_data.sortBy(lambda x: x[0][1]).map(lambda x:x[0][0]).take(20))

		# print(top_20_feature)
		# print("#########################################")
		# print(bottom_20_feature)

		def calc_p_value(x):
			dof = len(x[1])-2
			rss = 0.0
			for i in range(len(x[1])):
				val = x[1][i][1]-(x[0][1]*x[1][i][0])
				rss += val *val
			s_square = rss/dof
			denominator = dof+2
			plt_beta = stats.t.cdf(x[0][1]/math.sqrt((s_square/denominator)),dof)
			if plt_beta <0.5:
				return ((x[0][0]),(x[0][1],2*plt_beta*1000))
			else:
				return ((x[0][0]),(x[0][1],(1-plt_beta)*2*1000))


		pvalue_20_pos_cor = standardize_data.filter(lambda x: x[0][0] in brdcast_top_20_feature.value).map(lambda x: calc_p_value(x))
		pvalue_20_neg_cor = standardize_data.filter(lambda x: x[0][0] in brdcast_bottom_20_feature.value).map(lambda x: calc_p_value(x))

		print(" For goal ##",goal[i])
		print("#########################################")
		print()
		print("Top 20 most important feature ")
		print()
		print(pvalue_20_pos_cor.collect())
		print()
		print("#########################################")
		print()
		print("Bottom 20 most important feature ")
		print()
		print(pvalue_20_neg_cor.collect())
		print()


if __name__ == "__main__":
	main(sys.argv)


