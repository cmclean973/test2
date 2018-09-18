import pandas as pd
import numpy as np

def Recover(cum):
                
				def Drawdown(cum):
					peak=np.maximum.accumulate(cum)
					ddx=cum/peak
					return ddx

				def Rec(cum1):
					DD = Drawdown(cum1)-1
					#This looks like grouped by max price on an accumulating basis, but then equal to min price (?)
					Allpeakper = cum1.groupby(np.maximum.accumulate(cum1)).min()
					#Find Dates corresponding to peak
					Allpeakdates = cum1.index[cum1.isin(Allpeakper)]
					peakper = Allpeakper[Allpeakper != Allpeakper.index]
					DDper = DD[cum1.isin(peakper)]
					DDperlen = cum1.groupby(np.maximum.accumulate(cum1)).count()[Allpeakper != Allpeakper.index]
					DDtroughdates = DDper.index
					DDstartdates = pd.Series(cum1.index[cum1.isin(cum1.groupby(np.maximum.accumulate(cum1)).first())][Allpeakper != Allpeakper.index],index=DDper.index)
					DDenddates = pd.Series(cum1.index[cum1.isin(cum1.groupby(np.maximum.accumulate(cum1)).last())][Allpeakper != Allpeakper.index],index=DDper.index)
					Allres = pd.DataFrame([DDper.values,DDperlen.values,DDper.index.date,DDstartdates.apply(lambda x:x.date()).values,DDenddates.apply(lambda x:x.date()).values],index=[cum1.name+" DD",cum1.name+" Length",cum1.name+" Trough Date",cum1.name+" Start",cum1.name+" End"]).T
					#zeromult=np.where(DD==1,0,1)
					#cumrec=np.cumsum(zeromult)
					#timetorec=cumrec*zeromult
					return Allres
					
                #if more than one price series, run seperately for each
				if cum.ndim > 1:
					Recs = pd.concat([Rec(cum[x]) for x in cum.columns],axis=1)
				else:
					Recs = Rec(cum)
				return Recs

File_location = "C:\\Users\\cmclean\\Documents\\drawdown_price_data.xlsx"
input_file=pd.read_excel(File_location,[0])
price_data=input_file[0]
test = Recover(price_data)		