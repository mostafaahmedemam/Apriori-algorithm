import pandas as pd
import numpy as np
from itertools import combinations



def first_frequent_itemset(df , mini_support): #return level one item set 
    """
    This function determine the first itemset from the attributes that has support > minimum support 
    input : 
          df : a dataframe that contain the dataset 
          mini_support : an integer that represent the minimum support value

    output : 
           groups_list : a list that contains the first itemset 
    """

    groups_list = []
    
    df_rows , df_columns = df.shape
    for  label, content in df.items():
        
        if (content.nunique()==2 and sort(content.unique()) == [0,1]):
            content = content[content ==1].count()/float(df_rows)
            if (content >= mini_support): 
               groups_list.append(label)
               
        else:
            result= content.value_counts()
            result = result / float(df_rows)
            result = result[result>= mini_support]
            if result.size > 0:
                new_index = list((label+'_'+str(item)) for item in result.index.tolist())
                groups_list = groups_list+new_index
    
    return  groups_list



def support(data,comb, min_support):   
  """                                                                                                       
  input: data-> data frame contain data set
         comb-> list of list has combination of attributes :: (comb list)----->>>[["MOPLMIDD_1" ,"MOPLLAAG_5" ,"MBERZELF_6", "MBERBOER_8" ],
                                                                                  ["MOPLMIDD_5" ,"MOPLLAAG_4" ,"MBERZELF_2", "MBERBOER_3"]]
         min_support
  output: list of tuples contain (combination of attributes has support value above min support )
  
  description : check support for every list in (comb) list , if support above min support store output in list of tuples (every tuple has 
  attributes and support value)     
  """
  
  sucess_comb=[]
  success_support=[]
  
  for itt in range(0,len(comb)):
    com=list()
    
    filter=True
    for it in range(0,len(comb[itt])):
      
      
      both=comb[itt][it]
      parts=both.split('_')
      if len(parts)==1 :
        filter=(data[parts]==int(1)) & filter
      elif len(parts)==2:
        col_name=parts[0]
        attri_num=parts[1]
        filter=(data[col_name]==int(attri_num)) & filter
      com.append(both)
      
    select=data.loc[filter]
    support=len(select.index)/len(data.index)
    
    if support > min_support:
      sucess_comb.append(com)
      success_support.append(support)

  val=list(zip(sucess_comb,success_support))
  support_frame=pd.DataFrame(val)

  return support_frame,sucess_comb





def confidence(data,relations,min_confidance,min_support):
  """
  input:  data-> data frame contain data set
          relation :  list of lists contain final rules that want to test their confidance----->[[MSKC_2,MOPLLAAG_4 -> MOPLMIDD_3]]
          min confidance
          min support
  output: list of tuples contatin (rules that confidance above min confidance,confidance of rule,lift , laverage )

  description : check confidance for every rule in list(relations) by calling support function to calculate confidance value ,lift and leverage
  then if confidance value above min confidance then store date in list of tuples that conatian(rule,confidance,lift,leverage)

  hint: conf = support(MSKC_2,MOPLLAAG_4 , MOPLMIDD_3)/support(MSKC_2,MOPLLAAG_4)
        lift = conf/support(MOPLMIDD_3)
        leverage = support(MSKC_2,MOPLLAAG_4 , MOPLMIDD_3) -(support(MSKC_2,MOPLLAAG_4) * support(MOPLMIDD_3))
  """

  conf_list=[]
  success_relation=[]
  lift_list=[]
  leverage_list=[]
  for rel in range(0,len(relations)):
    
    x,y=relations[rel].split('->')
    left_side=(x.split(','))
    right_side=(y.split(','))
    all_side=left_side+right_side
    support_all,_=support(data,[all_side],min_support)
    support_all=support_all.at[0, 1]
    support_left,_=support(data,[left_side],min_support)
    support_left=support_left.at[0, 1]
    support_right,_=support(data,[right_side],min_support)
    support_right=support_right.at[0, 1]
    #printsupport_all.at[0, 1]))
    conf=support_all/support_left
    lift=conf/support_right
    leverage=support_all - (support_left * support_right)
      
    if conf > min_confidance:
      conf_list.append(conf) 
      success_relation.append(relations[rel])
      lift_list.append(lift)
      leverage_list.append(leverage)

  combine=list(zip(success_relation,conf_list,lift_list,leverage_list))
 
  return combine





def levels_combinations(l,level):
  '''
  This function compute the itemset for this level by making compinations of the itemset from the level below it and doing the prune step .
  input : 
        l : a list of list that contains the itemset from level below this level . ex [['A ,'B','C'], ['A,'D','C']]  means  ABC , ADC 
        level : an integer that represent the level number
   
  output :
        result : a list of list contains the itemset for this level  ex [['A','B', 'C','D'], ['A', 'N','M', 'H']]  means ABCD , ANMH

  '''
  
  com = list(combinations(l,level))
  result = []
  
  if level > 2:
      for i in range (len(com)):
          temp_list = []
          for j in range (len(com[i])):
              for k in range (len(com[i][j])):
                  temp_list.append(com[i][j][k])
          
          ind = 0
          for j in range (len(temp_list)):
              for k in range (j+1,len(temp_list)):
                  if '_' in temp_list[j] and '_' in temp_list[k] and (temp_list[k] !=temp_list[j]):
                      p1 = temp_list[j].find('_')
                      p2 = temp_list[k].find('_')
                      if temp_list[j][:p1] ==temp_list[k][:p2]:
                          ind=1
                          break 
              if ind ==1 :
                  break
          if ind ==0:
              x = set(temp_list)
              if len(x) == level:
                  result.append (list(x))
      return  result
  else:
      for i in range (len(com)):
          if '_' in com[i][0] and '_' in com[i][1] and (com[i][0] !=com[i][1]):
              p1 = com[i][0].find('_')
              p2 = com[i][1].find('_')
              if com[i][0][:p1] ==com[i][1][:p2]:
                  continue
          result.append([com[i][0],com[i][1]])
      return result




def combinations_rules(l):
  
  # this function compute all the possible rules from the last itemset . ex for ABC  A -> BC , B -> AC , C-> AB , AB -> C , AC -> B , BC -> A 
   #input :
    #      l: list of lists that contains the the last itemset   
   #output :
  #        result : list of list that contains all the possible rules  
  
    result= []
    for i in range (len(l)):
        uniuqe_items = set(l[i])
        for j in range (1,len(l[i])):
            comb=list(combinations(l[i],j))
            temp_list= []
            for k in range (len(comb)):
                temp_list.append(list(comb[k]))
            for m in range (len(temp_list)):
                rule= str()
                for n in range (len(temp_list[m])):
                    if (n ==len(temp_list[m])-1):
                        rule= rule + temp_list[m][n]
                    else:
                        rule= rule + temp_list[m][n]+','
                x = set(temp_list[m])
                rule = rule + '->'
                ind = 0
                for item in uniuqe_items - x:
                    if ind == len((uniuqe_items - x))-1:
                        rule = rule + item
                    else :
                        rule = rule + item +','
                    ind +=1
                result.append(rule)
    return result          
        


def main(min_support,min_confidance):
  
  labels=['MOPLMIDD' ,'MOPLLAAG' ,'MBERHOOG' ,'MBERZELF' ,'MBERBOER' ,'MBERMIDD' ,'MBERARBG' ,'MBERARBO' ,'MSKA' ,'MSKB1','MSKB2' ,'MSKC' ] 
  #load the dataset into a dataframe
  data = pd.read_csv('/content/drive/My Drive/ticdata2000.txt', sep="\t",names=labels, header=None,usecols=[*range(16,28)])
  
  
  #apply first_frequent_itemset function and get first itemset
  first_set=first_frequent_itemset(data, min_support)

  #apply levels_combination and return the itemset for the second level 
  comb=levels_combinations(first_set,2)
  if len(comb)==0 :
    print('NO associative rules')
  else:
    #every iteration we calculate support for the items in the current itemset and find the itemset for the next level from these items that has support > minimum support .
    #until we get the last itmeset 
    #this by applying the functions  : support , levels_combinations
    
    listt=[]
    for main_it in range(2,data.shape[1]):
      
      support_frame,list_success=support(data,comb,min_support)
      comb=levels_combinations(list_success,main_it+1)
      listt.append(list_success)
      if len(list_success)==0 :
        final_item_set=listt[len(listt)-2]
        break
    
    #apply combinations_rules function : to get all relations 
    relations_set=combinations_rules(final_item_set)
  
    #apply confidance func ->> parameter (data,final-item-set,min_confidance,min_support)
    #return list of tuples has all relations and their confidance,lift, leverage
    list_conf=confidence(data,relations_set,min_confidance,min_support)
    if len(list_conf)==0 :
      print('NO associative rules')
    else:
      for final_it in range(0,len(list_conf)):
        print('relation : '+list_conf[final_it][0])
        print('confidacne : '+str(list_conf[final_it][1]))
        print('lift :'+str(list_conf[final_it][2]))
        print('Leverage : ' + str(list_conf[final_it][3]))
        print('--------------------------------------------------')



#main(0.1,0.6) # test with min support 10% and confidacne 60%