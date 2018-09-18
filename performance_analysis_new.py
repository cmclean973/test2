#!\Users\efincham\Anaconda3\python.exe
import sys
import re
import numpy as np
import pandas as pd
import itertools
from datetime import datetime
import matplotlib.pyplot as plt
import pdb

#########################################################################################
#													  	  								#
#	Instructions for use:																#
#		1.	Ensure that the CSV file has a Date column ("DD/MM/YYYY")					#
#		2.	This script can only handle daily and monthly data							#
#		3.	Change the daily (3rd) command prompt argument accordingly					#
#		3.	Change directory on command prompt											#
#		4.	Type: performance_analysis.py <enter csv name>.csv <True or False>			#
#																						#
#########################################################################################

class analysis:
	def __init__(self, data, daily):
		self.daily = daily
		self.original = data
		self.original['Date'] = self.process_dates()
		''' Excess & Cumulative & Year Return '''
		self.excess_returns = self.excess_ret()
		self.cumulative = self.cumulative_ret()
		self.yearly_returns, self.yearly_relative = self.yearly_ret()
		''' Rolling Absolute & Relative & Annualised Yearly Return '''
		self.yearly_1_rolling, self.yearly_2_rolling, self.yearly_3_rolling, self.yearly_5_rolling, \
		self.yearly_1_relative_rolling, self.yearly_2_relative_rolling, self.yearly_3_relative_rolling, self.yearly_5_relative_rolling, \
		self.yearly_2_rolling_ann, self.yearly_3_rolling_ann, self.yearly_5_rolling_ann, \
		self.yearly_2_relative_rolling_ann, self.yearly_3_relative_rolling_ann, self.yearly_5_relative_rolling_ann = self.yearly_roll()
		''' Rolling Absolute & Relative Quarterly Return '''
		self.quarterly_returns, self.quarterly_relative = self.quarterly_ret()
		self.quarterly_rolling, self.quarterly_relative_rolling = self.quarterly_roll()
		''' Tracking Error & Information Ratio & Beta '''
		self.te_1_year_rolling, self.te_2_year_rolling, self.te_3_year_rolling, self.te_5_year_rolling = self.tracking_error()
		self.ir_1_year_rolling, self.ir_2_year_rolling, self.ir_3_year_rolling, self.ir_5_year_rolling = self.information_ratio()
		self.beta_1_year_rolling, self.beta_2_year_rolling, self.beta_3_year_rolling, self.beta_5_year_rolling = self.beta()
		pdb.set_trace()
		self.plot()

	def process_dates(self):
		return [datetime.strptime(row[1]['Date'], "%d/%m/%Y")for row in self.original.iterrows()]

	def excess_ret(self):
		column = [col for col in self.original.columns if (col != 'Date')]
		combos =  [list(col) for col in list(itertools.product(*[column, column])) if (col[0] != col[1])]
		excess_ret = [pd.DataFrame(self.original[combo[0]] - self.original[combo[1]], index = self.original.index, columns = ["|".join(combo)]) for combo in combos]
		excess_ret.insert(0, self.original['Date'])
		return pd.concat(excess_ret, axis = 1)

	def cumulative_ret(self):
		cumul_list = [self.cumulative_calc(self.original[col]) for col in [col_name for col_name in self.original.columns if col_name != 'Date']]
		cumul = pd.concat(cumul_list, axis = 1, keys = [s.name for s in cumul_list])
		dates = self.original['Date'].tolist(); dates.insert(0, None)
		return pd.concat([pd.DataFrame(dates, columns = ['Date']), cumul], axis = 1)

	def cumulative_calc(self, data):
		return_index = range(0, len(data) + 1)
		cumul_df = pd.Series(index = return_index); cumul_df[0] = 100.
		for c in range(1, len(cumul_df)):
			cumul_df[c] = cumul_df[c - 1.] + (cumul_df[c - 1.] * data[c - 1.])
		cumul_df.name = data.name
		return cumul_df

	def yearly_ret(self):
		''' Yearly Returns '''
		years = list(set([row.year for row in self.original['Date']]))
		years_dates = [row.year for row in self.original['Date']]
		yearly_df = pd.DataFrame(index = range(0, len(years)), columns = self.original.columns); yearly_df['Date'] = years
		for i, col in enumerate([col_name for col_name in self.original.columns if col_name != 'Date']):
			yearly_df[col] = [self.year(self.original[col], years_dates, yr) for yr in years]
		''' Yearly Relative Returns '''
		column = [col for col in self.original.columns if (col != 'Date')]
		combos =  [list(col) for col in list(itertools.product(*[column, column])) if (col[0] != col[1])]
		yearly_rel = [pd.DataFrame(yearly_df[combo[0]] - yearly_df[combo[1]], index = yearly_df.index, columns = ["|".join(combo)]) for combo in combos]
		yearly_rel.insert(0, yearly_df['Date'])
		return yearly_df, pd.concat(yearly_rel, axis = 1)

	def year(self, data, dates, date):
		dates = np.array(dates) == date
		val = data.values
		return np.prod(1. + val[dates]) - 1.

	def yearly_roll(self):
		if self.daily:
			wind = 252
		else:
			wind = 12
		''' Calculate Yearly Rolling Returns '''
		column = [col for col in self.original.columns if (col != 'Date')]
		roll_1 = [pd.DataFrame(self.original[col][1:].rolling(window = wind).apply(lambda x: np.prod(1. + x) - 1), columns = [col]) for col in column]; roll_1.insert(0, self.original['Date'])
		roll_2 = [pd.DataFrame(self.original[col][1:].rolling(window = 2 * wind).apply(lambda x: np.prod(1. + x) - 1), columns = [col]) for col in column]; roll_2.insert(0, self.original['Date'])
		roll_3 = [pd.DataFrame(self.original[col][1:].rolling(window = 3 * wind).apply(lambda x: np.prod(1. + x) - 1), columns = [col]) for col in column]; roll_3.insert(0, self.original['Date'])
		roll_5 = [pd.DataFrame(self.original[col][1:].rolling(window = 5 * wind).apply(lambda x: np.prod(1. + x) - 1), columns = [col]) for col in column]; roll_5.insert(0, self.original['Date'])
		''' Calculate Yearly Rolling Relative Returns '''
		column = [col for col in self.excess_returns.columns if (col != 'Date')]
		roll_rel_1 = [pd.DataFrame(self.excess_returns[col][1:].rolling(window = wind).apply(lambda x: np.prod(1. + x) - 1), columns = [col]) for col in column]; roll_rel_1.insert(0, self.excess_returns['Date'])
		roll_rel_2 = [pd.DataFrame(self.excess_returns[col][1:].rolling(window = 2 * wind).apply(lambda x: np.prod(1. + x) - 1), columns = [col]) for col in column]; roll_rel_2.insert(0, self.excess_returns['Date'])
		roll_rel_3 = [pd.DataFrame(self.excess_returns[col][1:].rolling(window = 3 * wind).apply(lambda x: np.prod(1. + x) - 1), columns = [col]) for col in column]; roll_rel_3.insert(0, self.excess_returns['Date'])
		roll_rel_5 = [pd.DataFrame(self.excess_returns[col][1:].rolling(window = 5 * wind).apply(lambda x: np.prod(1. + x) - 1), columns = [col]) for col in column]; roll_rel_5.insert(0, self.excess_returns['Date'])
		''' Calculate Yearly Rolling Returns Annualised '''
		column = [col for col in pd.concat(roll_2, axis = 1).columns if (col != 'Date')]
		roll_2_ann = [(1 + pd.concat(roll_2, axis = 1)[col]) ** (wind / (wind * 2)) - 1 for col in column]; roll_2_ann.insert(0, self.original['Date'])
		roll_3_ann = [(1 + pd.concat(roll_3, axis = 1)[col]) ** (wind / (wind * 3)) - 1 for col in column]; roll_3_ann.insert(0, self.original['Date'])
		roll_5_ann = [(1 + pd.concat(roll_5, axis = 1)[col]) ** (wind / (wind * 5)) - 1 for col in column]; roll_5_ann.insert(0, self.original['Date'])
		''' Calculate Yearly Rolling Relative Returns Annualised '''
		column = [col for col in pd.concat(roll_rel_2, axis = 1).columns if (col != 'Date')]
		roll_rel_2_ann = [(1 + pd.concat(roll_rel_2, axis = 1)[col]) ** (wind / (wind * 2)) - 1 for col in column]; roll_rel_2_ann.insert(0, self.original['Date'])
		roll_rel_3_ann = [(1 + pd.concat(roll_rel_3, axis = 1)[col]) ** (wind / (wind * 3)) - 1 for col in column]; roll_rel_3_ann.insert(0, self.original['Date'])
		roll_rel_5_ann = [(1 + pd.concat(roll_rel_5, axis = 1)[col]) ** (wind / (wind * 5)) - 1 for col in column]; roll_rel_5_ann.insert(0, self.original['Date'])
		return pd.concat(roll_1, axis = 1), pd.concat(roll_2, axis = 1), pd.concat(roll_3, axis = 1), pd.concat(roll_5, axis = 1), pd.concat(roll_rel_1, axis = 1), pd.concat(roll_rel_2, axis = 1), pd.concat(roll_rel_3, axis = 1), pd.concat(roll_rel_5, axis = 1), pd.concat(roll_2_ann, axis = 1), pd.concat(roll_3_ann, axis = 1), pd.concat(roll_5_ann, axis = 1), pd.concat(roll_rel_2_ann, axis = 1), pd.concat(roll_rel_3_ann, axis = 1), pd.concat(roll_rel_5_ann, axis = 1)

	def quarterly_ret(self):
		''' Calculate Quarterly Returns '''
		quart_list = [self.quarter(self.original[['Date', col]], col) for col in [col_name for col_name in self.original.columns if col_name != 'Date']]
		quart_df = pd.concat(quart_list, axis = 1); quart_df.index.name = 'Date'
		quart_df.reset_index(level = 0, inplace = True)
		''' Calculate Quarterly Relative Returns '''
		column = [col for col in self.original.columns if (col != 'Date')]
		combos =  [list(col) for col in list(itertools.product(*[column, column])) if (col[0] != col[1])]
		quart_rel = [pd.DataFrame(quart_df[combo[0]] - quart_df[combo[1]], index = quart_df.index, columns = ["|".join(combo)]) for combo in combos]
		quart_rel.insert(0, quart_df['Date'])
		return quart_df, pd.concat(quart_rel, axis = 1)

	def quarter(self, data, col):
		years = list(set([row.year for row in data['Date']])); quart_df, quart_dt = [], []
		year_list = [row.year for row in data['Date']]
		quarter_list = [i for row in data['Date'] for i, val in enumerate([[3, 2, 1], [6, 5, 4], [9, 8, 7], [12, 11, 10]]) if (row.month in val)]
		''' List Comprehension using Booleans to filter by Year then Quarter and returns the product of the result, looping over Years and Quarters '''
		quart_data = [np.prod(1. + data[col].values[np.array(year_list) == year][np.array(quarter_list)[np.array(year_list) == year] == i]) - 1. for year in years for i in range(0, 4) if len(data[col].values[np.array(year_list) == year][np.array(quarter_list)[np.array(year_list) == year] == i]) > 2]
		if self.daily:
			quart_date = [datetime.utcfromtimestamp(max(data['Date'].values[np.array(year_list) == year][np.array(quarter_list)[np.array(year_list) == year] == i]).astype('O') / 1e9) for year in years for i in range(0,4) if len(data['Date'].values[np.array(year_list) == year][np.array(quarter_list)[np.array(year_list) == year] == i]) > 62]
		else:
			quart_date = [datetime.utcfromtimestamp(max(data['Date'].values[np.array(year_list) == year][np.array(quarter_list)[np.array(year_list) == year] == i]).astype('O') / 1e9) for year in years for i in range(0,4) if len(data['Date'].values[np.array(year_list) == year][np.array(quarter_list)[np.array(year_list) == year] == i]) > 2]
		return pd.DataFrame(quart_data, index = quart_date, columns = [col])

	def quarterly_roll(self):
		''' Calculate Quarterly Rolling Returns '''
		column = [col for col in self.quarterly_returns.columns if (col != 'Date')]
		roll_1 = [pd.DataFrame(self.quarterly_returns[col].rolling(window = 4).apply(lambda x: np.prod(1. + x) - 1), columns = [col]) for col in column]
		roll_1.insert(0, self.quarterly_returns['Date'])
		''' Calculate Quarterly Rolling Relative Returns '''
		column = [col for col in self.quarterly_relative.columns if (col != 'Date')]
		roll_rel_1 = [pd.DataFrame(self.quarterly_relative[col].rolling(window = 4).apply(lambda x: np.prod(1. + x) - 1), columns = [col]) for col in column]
		roll_rel_1.insert(0, self.quarterly_relative['Date'])
		return pd.concat(roll_1, axis = 1), pd.concat(roll_rel_1, axis = 1)

	def tracking_error(self):
		if self.daily:
			wind = 252
		else:
			wind = 12
		''' Tracking Error Rolling 1, 2, 3, 5 Year '''
		column = [col for col in self.excess_returns.columns if (col != 'Date')]
		te_roll_1 = [pd.DataFrame(self.excess_returns[col][1:].rolling(window = wind * 1).apply(lambda x: np.std(x, ddof = 1) * np.sqrt(wind)), columns = [col]) for col in column]; te_roll_1.insert(0, self.excess_returns['Date'])
		te_roll_2 = [pd.DataFrame(self.excess_returns[col][1:].rolling(window = wind * 2).apply(lambda x: np.std(x, ddof = 1) * np.sqrt(wind)), columns = [col]) for col in column]; te_roll_2.insert(0, self.excess_returns['Date'])
		te_roll_3 = [pd.DataFrame(self.excess_returns[col][1:].rolling(window = wind * 3).apply(lambda x: np.std(x, ddof = 1) * np.sqrt(wind)), columns = [col]) for col in column]; te_roll_3.insert(0, self.excess_returns['Date'])
		te_roll_5 = [pd.DataFrame(self.excess_returns[col][1:].rolling(window = wind * 5).apply(lambda x: np.std(x, ddof = 1) * np.sqrt(wind)), columns = [col]) for col in column]; te_roll_5.insert(0, self.excess_returns['Date'])
		return pd.concat(te_roll_1, axis = 1), pd.concat(te_roll_2, axis = 1), pd.concat(te_roll_3, axis = 1), pd.concat(te_roll_5, axis = 1)
	
	def information_ratio(self):
		''' Information Ratio 1, 2, 3, 5 Year '''
		column = [col for col in self.original.columns if (col != 'Date')]
		combos =  [list(col) for col in list(itertools.product(*[column, column])) if (col[0] != col[1])]
		info_ratio_1 = [pd.DataFrame((self.yearly_1_rolling[combo[0]] - self.yearly_1_rolling[combo[1]]) / self.te_1_year_rolling[combo[0] + "|" + combo[1]], index = self.yearly_1_rolling.index, columns = ["|".join(combo)]) for combo in combos]; info_ratio_1.insert(0, self.excess_returns['Date'])
		info_ratio_2 = [pd.DataFrame((self.yearly_2_rolling_ann[combo[0]] - self.yearly_2_rolling_ann[combo[1]]) / self.te_2_year_rolling[combo[0] + "|" + combo[1]], index = self.yearly_2_rolling_ann.index, columns = ["|".join(combo)]) for combo in combos]; info_ratio_2.insert(0, self.excess_returns['Date'])
		info_ratio_3 = [pd.DataFrame((self.yearly_3_rolling_ann[combo[0]] - self.yearly_3_rolling_ann[combo[1]]) / self.te_3_year_rolling[combo[0] + "|" + combo[1]], index = self.yearly_3_rolling_ann.index, columns = ["|".join(combo)]) for combo in combos]; info_ratio_3.insert(0, self.excess_returns['Date'])
		info_ratio_5 = [pd.DataFrame((self.yearly_5_rolling_ann[combo[0]] - self.yearly_5_rolling_ann[combo[1]]) / self.te_5_year_rolling[combo[0] + "|" + combo[1]], index = self.yearly_5_rolling_ann.index, columns = ["|".join(combo)]) for combo in combos]; info_ratio_5.insert(0, self.excess_returns['Date'])
		return pd.concat(info_ratio_1, axis = 1), pd.concat(info_ratio_2, axis = 1), pd.concat(info_ratio_3, axis = 1), pd.concat(info_ratio_5, axis = 1)

	def beta(self):
		if self.daily:
			wind = 252
		else:
			wind = 12
		column = [col for col in self.original.columns if (col != 'Date')]
		combos =  [list(col) for col in list(itertools.product(*[column, column])) if (col[0] != col[1])]
		''' Covariance and Variance 1, 2, 3, 5 Year '''
		covar_1 = [pd.DataFrame(self.original[combo[0]][1:].rolling(window = wind * 1).cov(self.original[combo[1]][1:].rolling(window = wind * 1), pairwise = True), index = self.original.index, columns = ["|".join(combo)]) for combo in combos]
		var_1 = [pd.DataFrame(self.original[col][1:].rolling(window = wind * 1).var(), index = self.original.index, columns = [col]) for col in column]
		covar_2 = [pd.DataFrame(self.original[combo[0]][1:].rolling(window = wind * 2).cov(self.original[combo[1]][1:].rolling(window = wind * 2), pairwise = True), index = self.original.index, columns = ["|".join(combo)]) for combo in combos]
		var_2 = [pd.DataFrame(self.original[col][1:].rolling(window = wind * 2).var(), index = self.original.index, columns = [col]) for col in column]
		covar_3 = [pd.DataFrame(self.original[combo[0]][1:].rolling(window = wind * 3).cov(self.original[combo[1]][1:].rolling(window = wind * 3), pairwise = True), index = self.original.index, columns = ["|".join(combo)]) for combo in combos]
		var_3 = [pd.DataFrame(self.original[col][1:].rolling(window = wind * 3).var(), index = self.original.index, columns = [col]) for col in column]
		covar_5 = [pd.DataFrame(self.original[combo[0]][1:].rolling(window = wind * 5).cov(self.original[combo[1]][1:].rolling(window = wind * 5), pairwise = True), index = self.original.index, columns = ["|".join(combo)]) for combo in combos]
		var_5 = [pd.DataFrame(self.original[col][1:].rolling(window = wind * 5).var(), index = self.original.index, columns = [col]) for col in column]
		covar_1_year = pd.concat(covar_1, axis = 1); covar_2_year = pd.concat(covar_2, axis = 1); covar_3_year = pd.concat(covar_3, axis = 1); covar_5_year = pd.concat(covar_5, axis = 1)
		var_1_year = pd.concat(var_1, axis = 1); var_2_year = pd.concat(var_2, axis = 1); var_3_year = pd.concat(var_3, axis = 1); var_5_year = pd.concat(var_5, axis = 1)
		''' Beta 1, 2, 3, 5 Year '''
		beta_1_year = pd.concat([covar_1_year["|".join(combo)] / var_1_year[combo[1]] for combo in combos], axis = 1, keys = ["|".join(combo) for combo in combos]); beta_1_year.index = self.original['Date']; beta_1_year.reset_index(level = 0, inplace = True)
		beta_2_year = pd.concat([covar_2_year["|".join(combo)] / var_2_year[combo[1]] for combo in combos], axis = 1, keys = ["|".join(combo) for combo in combos]); beta_2_year.index = self.original['Date']; beta_2_year.reset_index(level = 0, inplace = True)
		beta_3_year = pd.concat([covar_3_year["|".join(combo)] / var_3_year[combo[1]] for combo in combos], axis = 1, keys = ["|".join(combo) for combo in combos]); beta_3_year.index = self.original['Date']; beta_3_year.reset_index(level = 0, inplace = True)
		beta_5_year = pd.concat([covar_5_year["|".join(combo)] / var_5_year[combo[1]] for combo in combos], axis = 1, keys = ["|".join(combo) for combo in combos]); beta_5_year.index = self.original['Date']; beta_5_year.reset_index(level = 0, inplace = True)
		return beta_1_year, beta_2_year, beta_3_year, beta_5_year 
	
	def plot(self):
		cont = True
		dataframe_dict = {"self.excess_returns": self.excess_returns, "self.cumulative" : self.cumulative, "self.yearly_returns": self.yearly_returns, "self.yearly_relative": self.yearly_relative, "self.yearly_1_rolling": self.yearly_1_rolling, "self.yearly_2_rolling": self.yearly_2_rolling, "self.yearly_3_rolling": self.yearly_3_rolling, "self.yearly_5_rolling": self.yearly_5_rolling, "self.yearly_1_relative_rolling": self.yearly_1_relative_rolling, "self.yearly_2_relative_rolling": self.yearly_2_relative_rolling, "self.yearly_3_relative_rolling": self.yearly_3_relative_rolling, "self.yearly_5_relative_rolling": self.yearly_5_relative_rolling, "self.yearly_2_rolling_ann": self.yearly_2_rolling_ann, "self.yearly_3_rolling_ann": self.yearly_3_rolling_ann, "self.yearly_5_rolling_ann": self.yearly_5_rolling_ann, "self.yearly_2_relative_rolling_ann": self.yearly_2_relative_rolling_ann, "self.yearly_3_relative_rolling_ann": self.yearly_3_relative_rolling_ann, "self.yearly_5_relative_rolling_ann": self.yearly_5_relative_rolling_ann, "self.quarterly_returns": self.quarterly_returns, "self.quarterly_relative": self.quarterly_relative, "self.quarterly_rolling": self.quarterly_rolling, "self.quarterly_relative_rolling": self.quarterly_relative_rolling, "self.te_1_year_rolling": self.te_1_year_rolling, "self.te_2_year_rolling": self.te_2_year_rolling, "self.te_3_year_rolling": self.te_3_year_rolling, "self.te_5_year_rolling": self.te_5_year_rolling, "self.ir_1_year_rolling": self.ir_1_year_rolling, "self.ir_2_year_rolling": self.ir_2_year_rolling, "self.ir_3_year_rolling": self.ir_3_year_rolling, "self.ir_5_year_rolling": self.ir_5_year_rolling, "self.beta_1_year_rolling": self.beta_1_year_rolling, "self.beta_2_year_rolling": self.beta_2_year_rolling, "self.beta_3_year_rolling": self.beta_3_year_rolling, "self.beta_5_year_rolling": self.beta_5_year_rolling}
		dataframe_list = [key for key in dataframe_dict]
		print("##################################################################################")
		print("                                       Plot                                       ")	
		print("##################################################################################","\n")
		print("-- Absolute Columns --")
		print(", ".join(self.original[[col for col in self.original.columns if (col != 'Date')]]),"\n")
		print("-- Relative Columns --")
		print(", ".join(self.excess_returns[[col for col in self.excess_returns.columns if (col != 'Date')]]),"\n\n")
		print("##################################################################################","\n")
		print("-- DataFrame Names --")
		print("\n".join(dataframe_list),"\n")
		print("##################################################################################","\n")
		print('To Plot: <series_name_1><["column_name"]>||<series_name_2><["column_name"]>||<bar or line>',"\n")
		print("To Retrieve Column Names for a DataFrame: <series_name>","\n")
		print('Note that "" are essential',"\n")
		print("##################################################################################","\n")
		while cont:
			chart = input("Input: ")
			if chart.lower().strip() == "exit":
				cont = False
			elif len(chart.split("||")) == 1:
				columns = [col for col in dataframe_dict[chart].columns if col != 'Date']
				print("DataFrame Columns:", ", ".join(columns))
			elif len(chart.split("||")) == 3:
				targets, kind = chart.split("||")[:2], chart.split("||")[2]
				request = list(zip([tar.split("[")[0] for tar in targets], [re.search(r'\[\"(.*)\"\]', tar).group(1) for tar in targets]))
				data = pd.concat([dataframe_dict[request[0][0]][request[0][1]], dataframe_dict[request[1][0]][request[1][1]]], axis = 1)
				data.index = dataframe_dict[request[0][0]]['Date']
				data.plot(title =  " vs. ".join(data.columns), kind = kind)
				plt.show()
			else:
				print('Try again. Input two DataFrames (with "column references") separated by "||" followed by chart type')
			
if __name__ == "__main__":
	performance = analysis(pd.read_csv(sys.argv[1]), sys.argv[2] == 'True')