#from NFL_utils import cleaner, condenser, player_pivot, feature_dict

# NFL BIG Data Processing Tools
# Cleaner - clean columns
# Dropper - Drop col to make relevant dataframe
# player_pivot - Pivot player details into one row per play
# feature_dict - reference dictionary to see original features

import pandas as pd
import numpy as np
import datetime
import tqdm
import random
import math


def cleaner(df_train,misc=True):
    #abbreviations
    df_train.loc[df_train.VisitorTeamAbbr == "ARI", 'VisitorTeamAbbr'] = "ARZ"
    df_train.loc[df_train.HomeTeamAbbr == "ARI", 'HomeTeamAbbr'] = "ARZ"

    df_train.loc[df_train.VisitorTeamAbbr == "BAL", 'VisitorTeamAbbr'] = "BLT"
    df_train.loc[df_train.HomeTeamAbbr == "BAL", 'HomeTeamAbbr'] = "BLT"

    df_train.loc[df_train.VisitorTeamAbbr == "CLE", 'VisitorTeamAbbr'] = "CLV"
    df_train.loc[df_train.HomeTeamAbbr == "CLE", 'HomeTeamAbbr'] = "CLV"

    df_train.loc[df_train.VisitorTeamAbbr == "HOU", 'VisitorTeamAbbr'] = "HST"
    df_train.loc[df_train.HomeTeamAbbr == "HOU", 'HomeTeamAbbr'] = "HST"
    #important identifiers
    df_train['ToLeft'] = df_train.PlayDirection == "left"
    df_train['IsRusher'] = df_train.NflId == df_train.NflIdRusher   
    df_train['OffDef'] = 'Defense'
    df_train.loc[((df_train.Team == 'home')&(df_train.PossessionTeam==df_train.HomeTeamAbbr)) | ((df_train.Team == 'away') & (df_train.PossessionTeam == df_train.VisitorTeamAbbr)),'OffDef'] = 'Offense'
    df_train.loc[df_train['NflId']==df_train['NflIdRusher'],'OffDef'] ='Rusher'
    #standardize all angles
    df_train['TeamOnOffense'] = "home"
    df_train.loc[df_train.PossessionTeam != df_train.HomeTeamAbbr, 'TeamOnOffense'] = "away"
    df_train['IsOnOffense'] = df_train.Team == df_train.TeamOnOffense # Is player on offense?
    df_train['YardLine_std'] = 100 - df_train.YardLine
    df_train.loc[df_train.FieldPosition.fillna('') == df_train.PossessionTeam,  
              'YardLine_std'
             ] = df_train.loc[df_train.FieldPosition.fillna('') == df_train.PossessionTeam,  
              'YardLine']
    df_train['YardLine_grid'] = 110 - df_train.YardLine
    df_train.loc[df_train.FieldPosition.fillna('') == df_train.PossessionTeam,  
              'YardLine_grid'
             ] = df_train.loc[df_train.FieldPosition.fillna('') == df_train.PossessionTeam,  
              'YardLine']+10
    df_train['X_std'] = df_train.X
    df_train.loc[df_train.ToLeft, 'X_std'] = 120 - df_train.loc[df_train.ToLeft, 'X'] 
    df_train['Y_std'] = df_train.Y
    df_train.loc[df_train.ToLeft, 'Y_std'] = 160/3 - df_train.loc[df_train.ToLeft, 'Y']
    ##Standardize Orientation
    df_train['Orientation'] = np.mod(90 - df_train['Orientation'], 360) ## Correcting Orientation to face forward
    df_train['Orientation_std'] = df_train.Orientation
    df_train.loc[df_train.ToLeft, 'Orientation_std'] = np.mod(180 + df_train.loc[df_train.ToLeft, 'Orientation_std'], 360)
    ## Standardize Direction
    df_train['Dir'] = np.mod(90 - df_train['Dir'], 360) # Correcting Dir to face forward
    df_train['Dir_std'] = df_train.Dir
    df_train.loc[df_train.ToLeft, 'Dir_std'] = np.mod(180 + df_train.loc[df_train.ToLeft, 'Dir_std'], 360)
    #Correction of 2017 Orientation being 90 off.
    df_train.loc[df_train['Season'] == 2017, 'Orientation_std'] = np.mod(-90 + df_train.loc[df_train['Season'] == 2017, 'Orientation_std'], 360)
    #Direction Vectors
    df_train['S_dx'] = df_train.S*np.cos((df_train.Dir_std)*np.pi/180.0)
    df_train['S_dy'] = df_train.S*np.sin((df_train.Dir_std)*np.pi/180.0)
    df_train['A_dx'] = df_train.A*np.cos((df_train.Dir_std)*np.pi/180.0)
    df_train['A_dy'] = df_train.A*np.sin((df_train.Dir_std)*np.pi/180.0)
    #Standardizing Speed (2017 to everything else)
#     traindf.loc[traindf['Season'] == 2017, 'S'] = (traindf['S'][traindf['Season'] == 2017] - 2.4355) / 1.2930 * 1.4551 + 2.7570
    #PlayerHeight
    df_train['PlayerHeight'] = df_train['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))
    #Player Force
    df_train['Force']=df_train['PlayerWeight']*df_train['A']
    
####################### Creating a dataframe with only rusher rows to clean only game specific data #############################

    df_rusher =df_train.loc[df_train['NflId']==df_train['NflIdRusher']].copy()
    #Attribute Change
#     df_rusher['Down']=df_rusher['Down'].astype('object')
#     df_rusher['Quarter']=df_rusher['Quarter'].astype('object')
#     df_rusher['YardLine_std_ob']=df_rusher['YardLine_std'].astype('object')

    #Scoring and conversion of Home/Away to Actual Team Names
    df_rusher.loc[df_rusher['PossessionTeam']==df_rusher['VisitorTeamAbbr'],'OffenseHome'] = 0
    df_rusher.loc[df_rusher['PossessionTeam']==df_rusher['HomeTeamAbbr'],'OffenseHome'] = 1
    df_rusher.loc[df_rusher['OffenseHome']==1,'DefenseTeam']=df_rusher['VisitorTeamAbbr']
    df_rusher.loc[df_rusher['OffenseHome']==0,'DefenseTeam']=df_rusher['HomeTeamAbbr']
    df_rusher.loc[df_rusher['OffenseHome']==1,'OffScoreDiff']=df_rusher['HomeScoreBeforePlay']-df_rusher['VisitorScoreBeforePlay']
    df_rusher.loc[df_rusher['OffenseHome']==0,'OffScoreDiff']=df_rusher['VisitorScoreBeforePlay']-df_rusher['HomeScoreBeforePlay']
    #GameClock in Seconds
    def strtoseconds(txt):
        txt = txt.split(':')
        ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60
        return ans
    df_rusher['GameClock'] = df_rusher['GameClock'].apply(strtoseconds)
    #Time Elasped in Game in Minutes
    df_rusher.loc[df_rusher['Quarter']< 5,'TimeElapsed'] = ((df_rusher.loc[df_rusher['Quarter']< 5,'Quarter']-1.0)*900.0+(900.0-df_rusher.loc[df_rusher['Quarter']< 5,'GameClock']))/60.0
    df_rusher.loc[df_rusher['Quarter']==5,'TimeElapsed'] = ((df_rusher.loc[df_rusher['Quarter']==5,'Quarter']-1.0)*900.0+(600.0-df_rusher.loc[df_rusher['Quarter']==5,'GameClock']))/60.0
    #Conversion to DateTime
    df_rusher['TimeHandoff'] = df_rusher['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    df_rusher['TimeSnap'] = df_rusher['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    df_rusher['PlayerBirthDate'] = df_rusher['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))
    #Time Features
    seconds_in_year = 60*60*24*365.25
    df_rusher['TimeDelta'] = (df_rusher['TimeHandoff'] - df_rusher['TimeSnap']).map(lambda x:x.seconds)
    df_rusher['TimeDelta'] = df_rusher.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
    df_rusher['PlayerAge'] = df_rusher.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)
    # df_rusher['Year'] = df_rusher.apply(lambda row: row['TimeHandoff'].year, axis =1) #similar to season
    #df_rusher['Month'] = df_rusher.apply(lambda row: row['TimeHandoff'].month,axis=1)
    #df_rusher['Hour'] = df_rusher.apply(lambda row: row['TimeHandoff'].hour,axis =1)
    
    #Rusher Position Classification
    def position_indicator(x):
        if x == 'RB' or x == 'HB':
            return 'RB'
        elif x =='QB':
            return 'QB'
        else:
            return 'Trick'
    df_rusher['Position']=df_rusher['Position'].apply(position_indicator)
    
    #Offense Formation Classification
    def Offense_formation(x):
        if x == 'SINGLEBACK':
            return 'SINGLEBACK'
        elif x =='SHOTGUN':
            return 'SHOTGUN'
        elif x =='I_FORM':
            return 'I_FORM'
        elif x == 'PISTOL':
            return 'PISTOL'
        elif x == 'JUMBO':
            return 'JUMBO'
        elif x == 'WILDCAT':
            return 'WILDCAT'
        else:
            return 'OTHER'
    df_rusher['OffenseFormation']=df_rusher['OffenseFormation'].apply(Offense_formation)
    
    #Personnel Splits
    def split_personnel(s):
        splits = s.split(',')
        for i in range(len(splits)):
            splits[i] = splits[i].strip()

        return splits

    def defense_formation(l):
        dl = 0
        lb = 0
        db = 0
        other = 0

        for position in l:
            sub_string = position.split(' ')
            if sub_string[1] == 'DL':
                dl += int(sub_string[0])
            elif sub_string[1] in ['LB','OL']:
                lb += int(sub_string[0])
            else:
                db += int(sub_string[0])

        counts = (dl,lb,db,other)

        return counts

    def offense_formation(l):
        qb = 0
        rb = 0
        wr = 0
        te = 0
        ol = 0

        sub_total = 0
        qb_listed = False
        for position in l:
            sub_string = position.split(' ')
            pos = sub_string[1]
            cnt = int(sub_string[0])

            if pos == 'QB':
                qb += cnt
                sub_total += cnt
                qb_listed = True
            # Assuming LB is a line backer lined up as full back
            elif pos in ['RB','LB']:
                rb += cnt
                sub_total += cnt
            # Assuming DB is a defensive back and lined up as WR
            elif pos in ['WR','DB']:
                wr += cnt
                sub_total += cnt
            elif pos == 'TE':
                te += cnt
                sub_total += cnt
            # Assuming DL is a defensive lineman lined up as an additional line man
            else:
                ol += cnt
                sub_total += cnt

        # If not all 11 players were noted at given positions we need to make some assumptions
        # I will assume if a QB is not listed then there was 1 QB on the play
        # If a QB is listed then I'm going to assume the rest of the positions are at OL
        # This might be flawed but it looks like RB, TE and WR are always listed in the personnel
        if sub_total < 11:
            diff = 11 - sub_total
            if not qb_listed:
                qb += 1
                diff -= 1
            ol += diff

        counts = (qb,rb,wr,te,ol)

        return counts
    
    df_rusher['DefensePersonnel'] = df_rusher['DefensePersonnel'].apply(lambda x: split_personnel(x))
    df_rusher['DefensePersonnel'] = df_rusher['DefensePersonnel'].apply(lambda x: defense_formation(x))
    df_rusher['num_DL'] = df_rusher['DefensePersonnel'].apply(lambda x: x[0])
    df_rusher['num_LB'] = df_rusher['DefensePersonnel'].apply(lambda x: x[1])
    df_rusher['num_DB'] = df_rusher['DefensePersonnel'].apply(lambda x: x[2])
    
    df_rusher['OffensePersonnel'] = df_rusher['OffensePersonnel'].apply(lambda x: split_personnel(x))
    df_rusher['OffensePersonnel'] = df_rusher['OffensePersonnel'].apply(lambda x: offense_formation(x))
    df_rusher['num_QB'] = df_rusher['OffensePersonnel'].apply(lambda x: x[0])
    df_rusher['num_RB'] = df_rusher['OffensePersonnel'].apply(lambda x: x[1])
    df_rusher['num_WR'] = df_rusher['OffensePersonnel'].apply(lambda x: x[2])
    df_rusher['num_TE'] = df_rusher['OffensePersonnel'].apply(lambda x: x[3])
    df_rusher['num_OL'] = df_rusher['OffensePersonnel'].apply(lambda x: x[4])

    # Let's create some features to specify if the OL is covered
    df_rusher['OL_diff'] = df_rusher['num_OL'] - df_rusher['num_DL']
    df_rusher['OL_TE_diff'] = (df_rusher['num_OL'] + df_rusher['num_TE']) - df_rusher['num_DL']
    # Let's create a feature to specify if the defense is preventing the run
    # Let's just assume 7 or more DL and LB is run prevention
    df_rusher['run_def'] = (df_rusher['num_DL'] + df_rusher['num_LB'] > 6).astype(int)
    df_rusher.drop(['OffensePersonnel','DefensePersonnel'], axis=1, inplace=True)

    
    
############################################### MISC Cleaning Columns ###############################################
    if misc:
        #Game Weather
        df_rusher['GameWeather'] = df_rusher['GameWeather'].str.lower()
        indoor = "indoor"
        df_rusher['GameWeather'] = df_rusher['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)
        df_rusher['GameWeather'] = df_rusher['GameWeather'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)
        df_rusher['GameWeather'] = df_rusher['GameWeather'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
        df_rusher['GameWeather'] = df_rusher['GameWeather'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
        
        def give_me_GameWeather(x):
            x = str(x).lower()
            if 'indoor' in x:
                return  'indoor'
            elif 'cloud' in x or 'coudy' in x or 'clouidy' in x:
                return 'cloudy'
            elif 'rain' in x or 'shower' in x:
                return 'rain'
            elif 'sunny' in x:
                return 'sunny'
            elif 'clear' in x:
                return 'clear'
            elif 'cold' in x or 'cool' in x:
                return 'cool'
            elif 'snow' in x:
                return 'snow'
            return x
        df_rusher['GameWeather'] = df_rusher['GameWeather'].apply(give_me_GameWeather)
        
        
    ######## if we want to map to certain value #####
#         def map_weather(txt):
#             ans = 1
#             if pd.isna(txt):
#                 return 0
#             if 'partly' in txt:
#                 ans*=0.5
#             if 'climate controlled' in txt or 'indoor' in txt:
#                 return ans*3
#             if 'sunny' in txt or 'sun' in txt:
#                 return ans*2
#             if 'clear' in txt:
#                 return ans
#             if 'cloudy' in txt:
#                 return -ans
#             if 'rain' in txt or 'rainy' in txt:
#                 return -2*ans
#             if 'snow' in txt:
#                 return -3*ans
#             return 0        
        
        
        #Wind Speed
        df_rusher['WindSpeed'] = df_rusher['WindSpeed'].apply(lambda x: str(x).lower().replace('mph', '').strip() if not pd.isna(x) else x) # strip ones with mph
        df_rusher['WindSpeed'] = df_rusher['WindSpeed'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x) # had some x-y so take the avg
        df_rusher['WindSpeed'] = df_rusher['WindSpeed'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x) # had one with gust up to. avg it
        
        def str_to_float(txt):
            try:
                return float(txt)
            except:
                return -1

        df_rusher['WindSpeed'] = df_rusher['WindSpeed'].apply(str_to_float)
        
        #Wind Direction
        north = ['N','From S','North']

        south = ['S','From N','South','s']

        west = ['W','From E','West']

        east = ['E','From W','from W','EAST','East']

        north_east = ['FROM SW','FROM SSW','FROM WSW','NE','NORTH EAST','North East','East North East','NorthEast','Northeast','ENE','From WSW','From SW']
        north_west = ['E','From ESE','NW','NORTHWEST','N-NE','NNE','North/Northwest','W-NW','WNW','West Northwest','Northwest','NNW','From SSE']
        south_east = ['E','From WNW','SE','SOUTHEAST','South Southeast','East Southeast','Southeast','SSE','From SSW','ESE','From NNW']
        south_west = ['E','From ENE','SW','SOUTHWEST','W-SW','South Southwest','West-Southwest','WSW','SouthWest','Southwest','SSW','From NNE']
        no_wind = ['clear','Calm']
        nan = ['1','8','13']
        
        df_rusher['WindDirection'] = df_rusher['WindDirection'].replace(north,'north')
        df_rusher['WindDirection'] = df_rusher['WindDirection'].replace(south,'south')
        df_rusher['WindDirection'] = df_rusher['WindDirection'].replace(west,'west')
        df_rusher['WindDirection'] = df_rusher['WindDirection'].replace(east,'east')
        df_rusher['WindDirection'] = df_rusher['WindDirection'].replace(north_east,'north_east')
        df_rusher['WindDirection'] = df_rusher['WindDirection'].replace(north_west,'north_west')
        df_rusher['WindDirection'] = df_rusher['WindDirection'].replace(south_east,'clear')
        df_rusher['WindDirection'] = df_rusher['WindDirection'].replace(south_west,'south_west')
        df_rusher['WindDirection'] = df_rusher['WindDirection'].replace(no_wind,'no_wind')
        df_rusher['WindDirection'] = df_rusher['WindDirection'].replace(nan,np.nan)
        
        #Stadium Type clean up
        outdoor = ['Outdoor', 'Outdoors', 'Cloudy', 'Heinz Field', 'Outdor', 'Ourdoor', 
                   'Outside', 'Outddors','Outdoor Retr Roof-Open', 'Oudoor', 'Bowl']

        indoor_closed = ['Indoors', 'Indoor', 'Indoor, Roof Closed', 'Indoor, Roof Closed', 'Retractable Roof',
                         'Retr. Roof-Closed', 'Retr. Roof - Closed', 'Retr. Roof Closed']

        indoor_open   = ['Indoor, Open Roof', 'Open', 'Retr. Roof-Open', 'Retr. Roof - Open']
        dome_closed   = ['Dome', 'Domed, closed', 'Closed Dome', 'Domed', 'Dome, closed']
        dome_open     = ['Domed, Open', 'Domed, open']

        df_rusher['StadiumType'] = df_rusher['StadiumType'].replace(outdoor,'outdoor')
        df_rusher['StadiumType'] = df_rusher['StadiumType'].replace(indoor_closed,'indoor_closed')
        df_rusher['StadiumType'] = df_rusher['StadiumType'].replace(indoor_open,'indoor_open')
        df_rusher['StadiumType'] = df_rusher['StadiumType'].replace(dome_closed,'dome_closed')
        df_rusher['StadiumType'] = df_rusher['StadiumType'].replace(dome_open,'dome_open')
        
        # Turf Clean up
        natural_grass = ['natural grass','Naturall Grass','Natural Grass']
        grass = ['Grass']

        fieldturf = ['FieldTurf','Field turf','FieldTurf360','Field Turf']

        artificial = ['Artificial','Artifical']

        df_rusher['Turf'] = df_rusher['Turf'].replace(natural_grass,'natural_grass')
        df_rusher['Turf'] = df_rusher['Turf'].replace(grass,'grass')
        df_rusher['Turf'] = df_rusher['Turf'].replace(fieldturf,'fieldturf')
        df_rusher['Turf'] = df_rusher['Turf'].replace(artificial,'artificial')
        
        #Temperature and Humidity FFill
        df_rusher['Humidity'].fillna(method='ffill', inplace=True)
        df_rusher['Temperature'].fillna(method='ffill', inplace=True)
        print('Preprocessed')
    return df_train,df_rusher

def rusher_features(df_rusher):
    #Distance from Line of Scrimmage
    df_rusher['back_from_scrimmage']=df_rusher['YardLine_grid']-df_rusher['X_std']
    df_rusher['back_from_1st']=df_rusher['back_from_scrimmage']+df_rusher['Distance']
    #Orientation towards downfield or not
    def back_direction(orientation):
        if orientation > 90.0 and orientation < 270.0:
            return 1
        else:
            return 0
    df_rusher['back_oriented_downfield']=df_rusher['Orientation_std'].apply(lambda x: back_direction(x))
    df_rusher['back_dir_downfield']=df_rusher['Dir_std'].apply(lambda x: back_direction(x))
    
    return df_rusher


def player_features(df_train):
    df_train.reset_index(inplace=True)
    df_train.drop('index', axis=1, inplace=True)
    #Creating Rusher Stats
    df_train["DistToRusher"] = np.sqrt((df_train.X_std - df_train.X_std[df_train.groupby('PlayId')['IsRusher'].transform('idxmax')].reset_index(drop=True))**2 \
                        +(df_train.Y_std - df_train.Y_std[df_train.groupby('PlayId')['IsRusher'].transform('idxmax')].reset_index(drop=True))**2)
    df_train['TimeToRusher']=df_train['DistToRusher']/df_train['S']
    df_train.loc[df_train['TimeToRusher']== float('inf'),'TimeToRusher']= df_train['TimeToRusher'].median()
    df_train['DistPastScrimmage']=df_train.YardLine_grid-df_train.X_std
    df_train['DistForwardScrimmage']=df_train.X_std-df_train.YardLine_grid
    
#     df_train['5yds_to_Rusher']=False
#     df_train.loc[df_train['DistToRusher']<5,'5yds_to_Rusher']=True
    
#     df_train['10yds_to_Rusher']=False
#     df_train.loc[df_train['DistToRusher']<10,'10yds_to_Rusher']=True
    
    
##################### Defense Player Indicators ######################
#     Def_1=df_train[df_train['OffDef']=='Defense'].iloc[::11]
#     Def_1['Def_1']=1
#     Def_1=Def_1[['PlayId','NflId','Def_1']]
#     df_train = pd.merge(df_train,Def_1,how='left',on=['PlayId','NflId'])
#     df_train['Def_1'].fillna(0,inplace=True);

#     Def_2=df_train[df_train['OffDef']=='Defense'].iloc[1::11]
#     Def_2['Def_2']=1
#     Def_2=Def_2[['PlayId','NflId','Def_2']]
#     df_train = pd.merge(df_train,Def_2,how='left',on=['PlayId','NflId'])
#     df_train['Def_2'].fillna(0,inplace=True);

#     Def_3=df_train[df_train['OffDef']=='Defense'].iloc[2::11]
#     Def_3['Def_3']=1
#     Def_3=Def_3[['PlayId','NflId','Def_3']]
#     df_train = pd.merge(df_train,Def_3,how='left',on=['PlayId','NflId'])
#     df_train['Def_3'].fillna(0,inplace=True);

#     Def_4=df_train[df_train['OffDef']=='Defense'].iloc[3::11]
#     Def_4['Def_4']=1
#     Def_4=Def_4[['PlayId','NflId','Def_4']]
#     df_train = pd.merge(df_train,Def_4,how='left',on=['PlayId','NflId'])
#     df_train['Def_4'].fillna(0,inplace=True);

#     Def_5=df_train[df_train['OffDef']=='Defense'].iloc[4::11]
#     Def_5['Def_5']=1
#     Def_5=Def_5[['PlayId','NflId','Def_5']]
#     df_train = pd.merge(df_train,Def_5,how='left',on=['PlayId','NflId'])
#     df_train['Def_5'].fillna(0,inplace=True);

#     Def_6=df_train[df_train['OffDef']=='Defense'].iloc[5::11]
#     Def_6['Def_6']=1
#     Def_6=Def_6[['PlayId','NflId','Def_6']]
#     df_train = pd.merge(df_train,Def_6,how='left',on=['PlayId','NflId'])
#     df_train['Def_6'].fillna(0,inplace=True);

#     Def_7=df_train[df_train['OffDef']=='Defense'].iloc[6::11]
#     Def_7['Def_7']=1
#     Def_7=Def_7[['PlayId','NflId','Def_7']]
#     df_train = pd.merge(df_train,Def_7,how='left',on=['PlayId','NflId'])
#     df_train['Def_7'].fillna(0,inplace=True);

#     Def_8=df_train[df_train['OffDef']=='Defense'].iloc[7::11]
#     Def_8['Def_8']=1
#     Def_8=Def_8[['PlayId','NflId','Def_8']]
#     df_train = pd.merge(df_train,Def_8,how='left',on=['PlayId','NflId'])
#     df_train['Def_8'].fillna(0,inplace=True);

#     Def_9=df_train[df_train['OffDef']=='Defense'].iloc[8::11]
#     Def_9['Def_9']=1
#     Def_9=Def_9[['PlayId','NflId','Def_9']]
#     df_train = pd.merge(df_train,Def_9,how='left',on=['PlayId','NflId'])
#     df_train['Def_9'].fillna(0,inplace=True);

#     Def_10=df_train[df_train['OffDef']=='Defense'].iloc[9::11]
#     Def_10['Def_10']=1
#     Def_10=Def_10[['PlayId','NflId','Def_10']]
#     df_train = pd.merge(df_train,Def_10,how='left',on=['PlayId','NflId'])
#     df_train['Def_10'].fillna(0,inplace=True);

#     Def_11=df_train[df_train['OffDef']=='Defense'].iloc[10::11]
#     Def_11['Def_11']=1
#     Def_11=Def_11[['PlayId','NflId','Def_11']]
#     df_train = pd.merge(df_train,Def_11,how='left',on=['PlayId','NflId'])
#     df_train['Def_11'].fillna(0,inplace=True);
  
    
############### Creating Distance between Defense ###############  

#     df_train["DistToDef_1"] = np.sqrt((df_train.X_std - df_train.X_std[df_train.groupby('PlayId')['Def_1'].transform('idxmax')].reset_index(drop=True))**2 \
#                         +(df_train.Y_std - df_train.Y_std[df_train.groupby('PlayId')['Def_1'].transform('idxmax')].reset_index(drop=True))**2)
#     df_train["DistToDef_2"] = np.sqrt((df_train.X_std - df_train.X_std[df_train.groupby('PlayId')['Def_2'].transform('idxmax')].reset_index(drop=True))**2 \
#                         +(df_train.Y_std - df_train.Y_std[df_train.groupby('PlayId')['Def_2'].transform('idxmax')].reset_index(drop=True))**2)
#     df_train["DistToDef_3"] = np.sqrt((df_train.X_std - df_train.X_std[df_train.groupby('PlayId')['Def_3'].transform('idxmax')].reset_index(drop=True))**2 \
#                         +(df_train.Y_std - df_train.Y_std[df_train.groupby('PlayId')['Def_3'].transform('idxmax')].reset_index(drop=True))**2)
#     df_train["DistToDef_4"] = np.sqrt((df_train.X_std - df_train.X_std[df_train.groupby('PlayId')['Def_4'].transform('idxmax')].reset_index(drop=True))**2 \
#                         +(df_train.Y_std - df_train.Y_std[df_train.groupby('PlayId')['Def_4'].transform('idxmax')].reset_index(drop=True))**2)
#     df_train["DistToDef_5"] = np.sqrt((df_train.X_std - df_train.X_std[df_train.groupby('PlayId')['Def_5'].transform('idxmax')].reset_index(drop=True))**2 \
#                         +(df_train.Y_std - df_train.Y_std[df_train.groupby('PlayId')['Def_5'].transform('idxmax')].reset_index(drop=True))**2)
#     df_train["DistToDef_6"] = np.sqrt((df_train.X_std - df_train.X_std[df_train.groupby('PlayId')['Def_6'].transform('idxmax')].reset_index(drop=True))**2 \
#                         +(df_train.Y_std - df_train.Y_std[df_train.groupby('PlayId')['Def_6'].transform('idxmax')].reset_index(drop=True))**2)
#     df_train["DistToDef_7"] = np.sqrt((df_train.X_std - df_train.X_std[df_train.groupby('PlayId')['Def_7'].transform('idxmax')].reset_index(drop=True))**2 \
#                         +(df_train.Y_std - df_train.Y_std[df_train.groupby('PlayId')['Def_7'].transform('idxmax')].reset_index(drop=True))**2)
#     df_train["DistToDef_8"] = np.sqrt((df_train.X_std - df_train.X_std[df_train.groupby('PlayId')['Def_8'].transform('idxmax')].reset_index(drop=True))**2 \
#                         +(df_train.Y_std - df_train.Y_std[df_train.groupby('PlayId')['Def_8'].transform('idxmax')].reset_index(drop=True))**2)
#     df_train["DistToDef_9"] = np.sqrt((df_train.X_std - df_train.X_std[df_train.groupby('PlayId')['Def_9'].transform('idxmax')].reset_index(drop=True))**2 \
#                         +(df_train.Y_std - df_train.Y_std[df_train.groupby('PlayId')['Def_9'].transform('idxmax')].reset_index(drop=True))**2)
#     df_train["DistToDef_10"] = np.sqrt((df_train.X_std - df_train.X_std[df_train.groupby('PlayId')['Def_10'].transform('idxmax')].reset_index(drop=True))**2 \
#                         +(df_train.Y_std - df_train.Y_std[df_train.groupby('PlayId')['Def_10'].transform('idxmax')].reset_index(drop=True))**2)
#     df_train["DistToDef_11"] = np.sqrt((df_train.X_std - df_train.X_std[df_train.groupby('PlayId')['Def_11'].transform('idxmax')].reset_index(drop=True))**2 \
#                         +(df_train.Y_std - df_train.Y_std[df_train.groupby('PlayId')['Def_11'].transform('idxmax')].reset_index(drop=True))**2)
    
        # Creating Features
################# Defense ###################    
    def_X_min=df_train.loc[df_train['OffDef']=='Defense'].groupby(['PlayId'])['X_std'].min().rename('def_X_min')
    FE = def_X_min.to_frame()
    FE['def_X_max']=df_train.loc[df_train['OffDef']=='Defense'].groupby(['PlayId'])['X_std'].max().rename('def_X_max')
    FE['def_X_avg']=df_train.loc[df_train['OffDef']=='Defense'].groupby(['PlayId'])['X_std'].mean().rename('def_X_avg')
    FE['def_X_std']=df_train.loc[df_train['OffDef']=='Defense'].groupby(['PlayId'])['X_std'].std().rename('def_X_std')
    # FE['def_X_sum']=df_train.loc[df_train['OffDef']=='Defense'].groupby(['PlayId'])['X_std'].sum().rename('def_X_sum')

    FE['def_Y_min']=df_train.loc[df_train['OffDef']=='Defense'].groupby(['PlayId'])['Y_std'].min().rename('def_Y_min')
    FE['def_Y_max']=df_train.loc[df_train['OffDef']=='Defense'].groupby(['PlayId'])['Y_std'].max().rename('def_Y_max')
    FE['def_Y_avg']=df_train.loc[df_train['OffDef']=='Defense'].groupby(['PlayId'])['Y_std'].mean().rename('def_Y_avg')
    FE['def_Y_std']=df_train.loc[df_train['OffDef']=='Defense'].groupby(['PlayId'])['Y_std'].std().rename('def_Y_std')

    FE['def_DistToRusher_min']=df_train.loc[df_train['OffDef']=='Defense'].groupby(['PlayId'])['DistToRusher'].min().rename('def_DistToRusher_min')
    FE['def_DistToRusher_max']=df_train.loc[df_train['OffDef']=='Defense'].groupby(['PlayId'])['DistToRusher'].max().rename('def_DistToRusher_max')
    FE['def_DistToRusher_avg']=df_train.loc[df_train['OffDef']=='Defense'].groupby(['PlayId'])['DistToRusher'].mean().rename('def_DistToRusher_avg')
    FE['def_DistToRusher_std']=df_train.loc[df_train['OffDef']=='Defense'].groupby(['PlayId'])['DistToRusher'].std().rename('def_DistToRusher_std')
    
    FE['def_TimeToRusher_min']=df_train.loc[df_train['OffDef']=='Defense'].groupby(['PlayId'])['TimeToRusher'].min().rename('def_TimeToRusher_min')
    FE['def_TimeToRusher_max']=df_train.loc[df_train['OffDef']=='Defense'].groupby(['PlayId'])['TimeToRusher'].max().rename('def_TimeToRusher_max')
    FE['def_TimeToRusher_avg']=df_train.loc[df_train['OffDef']=='Defense'].groupby(['PlayId'])['TimeToRusher'].mean().rename('def_TimeToRusher_avg')
    FE['def_TimeToRusher_std']=df_train.loc[df_train['OffDef']=='Defense'].groupby(['PlayId'])['TimeToRusher'].std().rename('def_TimeToRusher_std')
    
############ Closest 3 Defense #############
    closest_3_def = df_train[df_train['OffDef']=='Defense'].sort_values('DistToRusher').groupby('PlayId').head(3).sort_values('PlayId')
    
    FE['def_X_min_3']=closest_3_def.groupby(['PlayId'])['X_std'].min().rename('def_X_min_3')
    FE['def_X_max_3']=closest_3_def.groupby(['PlayId'])['X_std'].max().rename('def_X_max_3')
    FE['def_X_avg_3']=closest_3_def.groupby(['PlayId'])['X_std'].mean().rename('def_X_avg_3')
    FE['def_X_std_3']=closest_3_def.groupby(['PlayId'])['X_std'].std().rename('def_X_std_3')

    FE['def_Y_min_3']=closest_3_def.groupby(['PlayId'])['Y_std'].min().rename('def_Y_min_3')
    FE['def_Y_max_3']=closest_3_def.groupby(['PlayId'])['Y_std'].max().rename('def_Y_max_3')
    FE['def_Y_avg_3']=closest_3_def.groupby(['PlayId'])['Y_std'].mean().rename('def_Y_avg_3')
    FE['def_Y_std_3']=closest_3_def.groupby(['PlayId'])['Y_std'].std().rename('def_Y_std_3')
    
    FE['def_S_min_3']=closest_3_def.groupby(['PlayId'])['S'].min().rename('def_S_min_3')
    FE['def_S_max_3']=closest_3_def.groupby(['PlayId'])['S'].max().rename('def_S_max_3')
    FE['def_S_avg_3']=closest_3_def.groupby(['PlayId'])['S'].mean().rename('def_S_avg_3')
    FE['def_S_std_3']=closest_3_def.groupby(['PlayId'])['S'].std().rename('def_S_std_3')
    
    FE['def_A_min_3']=closest_3_def.groupby(['PlayId'])['A'].min().rename('def_A_min_3')
    FE['def_A_max_3']=closest_3_def.groupby(['PlayId'])['A'].max().rename('def_A_max_3')
    FE['def_A_avg_3']=closest_3_def.groupby(['PlayId'])['A'].mean().rename('def_A_avg_3')
    FE['def_A_std_3']=closest_3_def.groupby(['PlayId'])['A'].std().rename('def_A_std_3')
    

    FE['def_DistToRusher_min_3']=closest_3_def.groupby(['PlayId'])['DistToRusher'].min().rename('def_DistToRusher_min_3')
    FE['def_DistToRusher_max_3']=closest_3_def.groupby(['PlayId'])['DistToRusher'].max().rename('def_DistToRusher_max_3')
    FE['def_DistToRusher_avg_3']=closest_3_def.groupby(['PlayId'])['DistToRusher'].mean().rename('def_DistToRusher_avg_3')
    FE['def_DistToRusher_std_3']=closest_3_def.groupby(['PlayId'])['DistToRusher'].std().rename('def_DistToRusher_std_3')
    
    FE['def_TimeToRusher_min_3']=closest_3_def.groupby(['PlayId'])['TimeToRusher'].min().rename('def_TimeToRusher_min_3')
    FE['def_TimeToRusher_max_3']=closest_3_def.groupby(['PlayId'])['TimeToRusher'].max().rename('def_TimeToRusher_max_3')
    FE['def_TimeToRusher_avg_3']=closest_3_def.groupby(['PlayId'])['TimeToRusher'].mean().rename('def_TimeToRusher_avg_3')
    FE['def_TimeToRusher_std_3']=closest_3_def.groupby(['PlayId'])['TimeToRusher'].std().rename('def_TimeToRusher_std_3')
    
    
############### Defense that is 5 yds from Rusher ###############
#     FE['def_X_min_5']=df_train.loc[(df_train['OffDef']=='Defense')&(df_train['5yds_to_Rusher']==True)].groupby(['PlayId'])['X_std'].min().rename('def_X_min_5')
#     FE['def_X_max_5']=df_train.loc[(df_train['OffDef']=='Defense')&(df_train['5yds_to_Rusher']==True)].groupby(['PlayId'])['X_std'].max().rename('def_X_max_5')
#     FE['def_X_avg_5']=df_train.loc[(df_train['OffDef']=='Defense')&(df_train['5yds_to_Rusher']==True)].groupby(['PlayId'])['X_std'].mean().rename('def_X_avg_5')
#     FE['def_X_std_5']=df_train.loc[(df_train['OffDef']=='Defense')&(df_train['5yds_to_Rusher']==True)].groupby(['PlayId'])['X_std'].std().rename('def_X_std_5')
#     # FE['def_X_sum']=df_train.loc[df_train['OffDef']=='Defense'].groupby(['PlayId'])['X_std'].sum().rename('def_X_sum')
    
#     FE['def_Y_min_5']=df_train.loc[(df_train['OffDef']=='Defense')&(df_train['5yds_to_Rusher']==True)].groupby(['PlayId'])['Y_std'].min().rename('def_Y_min_5')
#     FE['def_Y_max_5']=df_train.loc[(df_train['OffDef']=='Defense')&(df_train['5yds_to_Rusher']==True)].groupby(['PlayId'])['Y_std'].max().rename('def_Y_max_5')
#     FE['def_Y_avg_5']=df_train.loc[(df_train['OffDef']=='Defense')&(df_train['5yds_to_Rusher']==True)].groupby(['PlayId'])['Y_std'].mean().rename('def_Y_avg_5')
#     FE['def_Y_std_5']=df_train.loc[(df_train['OffDef']=='Defense')&(df_train['5yds_to_Rusher']==True)].groupby(['PlayId'])['Y_std'].std().rename('def_Y_std_5')

#     FE['def_DistToRusher_min_5']=df_train.loc[(df_train['OffDef']=='Defense')&(df_train['5yds_to_Rusher']==True)].groupby(['PlayId'])['DistToRusher'].min().rename('def_DistToRusher_min_5')
#     FE['def_DistToRusher_max_5']=df_train.loc[(df_train['OffDef']=='Defense')&(df_train['5yds_to_Rusher']==True)].groupby(['PlayId'])['DistToRusher'].max().rename('def_DistToRusher_max_5')
#     FE['def_DistToRusher_avg_5']=df_train.loc[(df_train['OffDef']=='Defense')&(df_train['5yds_to_Rusher']==True)].groupby(['PlayId'])['DistToRusher'].mean().rename('def_DistToRusher_avg_5')
#     FE['def_DistToRusher_std_5']=df_train.loc[(df_train['OffDef']=='Defense')&(df_train['5yds_to_Rusher']==True)].groupby(['PlayId'])['DistToRusher'].std().rename('def_DistToRusher_std_5')
    
#     FE['def_TimeToRusher_min_5']=df_train.loc[(df_train['OffDef']=='Defense')&(df_train['5yds_to_Rusher']==True)].groupby(['PlayId'])['TimeToRusher'].min().rename('def_TimeToRusher_min_5')
#     FE['def_TimeToRusher_max_5']=df_train.loc[(df_train['OffDef']=='Defense')&(df_train['5yds_to_Rusher']==True)].groupby(['PlayId'])['TimeToRusher'].max().rename('def_TimeToRusher_max_5')
#     FE['def_TimeToRusher_avg_5']=df_train.loc[(df_train['OffDef']=='Defense')&(df_train['5yds_to_Rusher']==True)].groupby(['PlayId'])['TimeToRusher'].mean().rename('def_TimeToRusher_avg_5')
#     FE['def_TimeToRusher_std_5']=df_train.loc[(df_train['OffDef']=='Defense')&(df_train['5yds_to_Rusher']==True)].groupby(['PlayId'])['TimeToRusher'].std().rename('def_TimeToRusher_std_5')
    
#     FE['def_X_min_5'].fillna(0, inplace=True)
    # need to fill out rest of nans, this method can create a lot of nans if no one is in radius 

    
    ##### Penetration Metrics####
    FE['Def_past_scrimmage_yds']=df_train.loc[(df_train['OffDef']=='Defense')&(df_train['DistPastScrimmage']>0)].groupby(['PlayId'])['DistPastScrimmage'].sum().rename('Def_past_scrimmage_yds')
    FE['Def_past_scrimmage_yds'].fillna(0, inplace=True)
    FE['Def_past_scrimmage_count']=df_train.loc[(df_train['OffDef']=='Defense')&(df_train['DistPastScrimmage']>0)].groupby(['PlayId'])['DistPastScrimmage'].count().rename('Def_past_scrimmage_count')
    FE['Def_past_scrimmage_count'].fillna(0, inplace=True)

    
################## Offense ###############    
    FE['off_X_min']=df_train.loc[df_train['OffDef']=='Offense'].groupby(['PlayId'])['X_std'].min().rename('off_X_min')
    FE['off_X_max']=df_train.loc[df_train['OffDef']=='Offense'].groupby(['PlayId'])['X_std'].max().rename('off_X_max')
    FE['off_X_avg']=df_train.loc[df_train['OffDef']=='Offense'].groupby(['PlayId'])['X_std'].mean().rename('off_X_avg')
    FE['off_X_std']=df_train.loc[df_train['OffDef']=='Offense'].groupby(['PlayId'])['X_std'].std().rename('off_X_std')

    FE['off_Y_min']=df_train.loc[df_train['OffDef']=='Offense'].groupby(['PlayId'])['Y_std'].min().rename('off_Y_min')
    FE['off_Y_max']=df_train.loc[df_train['OffDef']=='Offense'].groupby(['PlayId'])['Y_std'].max().rename('off_Y_max')
    FE['off_Y_avg']=df_train.loc[df_train['OffDef']=='Offense'].groupby(['PlayId'])['Y_std'].mean().rename('off_Y_avg')
    FE['off_Y_std']=df_train.loc[df_train['OffDef']=='Offense'].groupby(['PlayId'])['Y_std'].std().rename('off_Y_std')

    FE['off_DistToRusher_min']=df_train.loc[df_train['OffDef']=='Offense'].groupby(['PlayId'])['DistToRusher'].min().rename('off_DistToRusher_min')
    FE['off_DistToRusher_max']=df_train.loc[df_train['OffDef']=='Offense'].groupby(['PlayId'])['DistToRusher'].max().rename('off_DistToRusher_max')
    FE['off_DistToRusher_avg']=df_train.loc[df_train['OffDef']=='Offense'].groupby(['PlayId'])['DistToRusher'].mean().rename('off_DistToRusher_avg')
    FE['off_DistToRusher_std']=df_train.loc[df_train['OffDef']=='Offense'].groupby(['PlayId'])['DistToRusher'].std().rename('off_DistToRusher_std')
    
    FE['Def_Spread_X']=FE['def_X_max']-FE['def_X_min']
    FE['Off_Spread_X']=FE['off_X_max']-FE['off_X_min']
    FE['Def_Spread_Y']=FE['def_Y_max']-FE['def_Y_min']
    FE['Off_Spread_Y']=FE['off_Y_max']-FE['off_Y_min']
    
############### Closest 3 Offense Exclude QB #############
    closest_3_off = df_train[(df_train['OffDef']=='Offense')&(df_train['Position']!='QB')].sort_values('DistToRusher').groupby('PlayId').head(3).sort_values('PlayId')
    
    FE['off_X_min_3']=closest_3_off.groupby(['PlayId'])['X_std'].min().rename('off_X_min_3')
    FE['off_X_max_3']=closest_3_off.groupby(['PlayId'])['X_std'].max().rename('off_X_max_3')
    FE['off_X_avg_3']=closest_3_off.groupby(['PlayId'])['X_std'].mean().rename('off_X_avg_3')
    FE['off_X_std_3']=closest_3_off.groupby(['PlayId'])['X_std'].std().rename('off_X_std_3')

    FE['off_Y_min_3']=closest_3_off.groupby(['PlayId'])['Y_std'].min().rename('off_Y_min_3')
    FE['off_Y_max_3']=closest_3_off.groupby(['PlayId'])['Y_std'].max().rename('off_Y_max_3')
    FE['off_Y_avg_3']=closest_3_off.groupby(['PlayId'])['Y_std'].mean().rename('off_Y_avg_3')
    FE['off_Y_std_3']=closest_3_off.groupby(['PlayId'])['Y_std'].std().rename('off_Y_std_3')
    
    FE['off_S_min_3']=closest_3_off.groupby(['PlayId'])['S'].min().rename('off_S_min_3')
    FE['off_S_max_3']=closest_3_off.groupby(['PlayId'])['S'].max().rename('off_S_max_3')
    FE['off_S_avg_3']=closest_3_off.groupby(['PlayId'])['S'].mean().rename('off_S_avg_3')
    FE['off_S_std_3']=closest_3_off.groupby(['PlayId'])['S'].std().rename('off_S_std_3')
    
    FE['off_A_min_3']=closest_3_off.groupby(['PlayId'])['A'].min().rename('off_A_min_3')
    FE['off_A_max_3']=closest_3_off.groupby(['PlayId'])['A'].max().rename('off_A_max_3')
    FE['off_A_avg_3']=closest_3_off.groupby(['PlayId'])['A'].mean().rename('off_A_avg_3')
    FE['off_A_std_3']=closest_3_off.groupby(['PlayId'])['A'].std().rename('off_A_std_3')

    FE['off_DistToRusher_min_3']=closest_3_off.groupby(['PlayId'])['DistToRusher'].min().rename('off_DistToRusher_min_3')
    FE['off_DistToRusher_max_3']=closest_3_off.groupby(['PlayId'])['DistToRusher'].max().rename('off_DistToRusher_max_3')
    FE['off_DistToRusher_avg_3']=closest_3_off.groupby(['PlayId'])['DistToRusher'].mean().rename('off_DistToRusher_avg_3')
    FE['off_DistToRusher_std_3']=closest_3_off.groupby(['PlayId'])['DistToRusher'].std().rename('off_DistToRusher_std_3')
    
#     FE['off_TimeToRusher_min_3']=closest_3_off.groupby(['PlayId'])['TimeToRusher'].min().rename('off_TimeToRusher_min_3')
#     FE['off_TimeToRusher_max_3']=closest_3_off.groupby(['PlayId'])['TimeToRusher'].max().rename('off_TimeToRusher_max_3')
#     FE['off_TimeToRusher_avg_3']=closest_3_off.groupby(['PlayId'])['TimeToRusher'].mean().rename('off_TimeToRusher_avg_3')
#     FE['off_TimeToRusher_std_3']=closest_3_off.groupby(['PlayId'])['TimeToRusher'].std().rename('off_TimeToRusher_std_3')
    
    
    
######################## All players #########################
#     FE['All_DistToRusher_min']=df_train.loc[df_train['OffDef']!='Rusher'].groupby(['PlayId'])['DistToRusher'].min().rename('off_DistToRusher_min')
#     FE['All_DistToRusher_max']=df_train.loc[df_train['OffDef']!='Rusher'].groupby(['PlayId'])['DistToRusher'].max().rename('off_DistToRusher_max')
#     FE['All_DistToRusher_avg']=df_train.loc[df_train['OffDef']!='Rusher'].groupby(['PlayId'])['DistToRusher'].mean().rename('off_DistToRusher_avg')
#     FE['All_DistToRusher_std']=df_train.loc[df_train['OffDef']!='Rusher'].groupby(['PlayId'])['DistToRusher'].std().rename('off_DistToRusher_std')
    
    
    
    FE.reset_index(inplace=True)
    
    return df_train,FE



def dropper(df):
    #Drop columns of least importance (Come back to add them later)
    Standardized = ['X','Y','Orientation','Dir','YardLine','PlayDirection','FieldPosition']
    FE_out = ['HomeScoreBeforePlay','VisitorScoreBeforePlay','OffenseHome','IsOnOffense',
              'PlayerBirthDate','TimeHandoff','TimeSnap']
    Temp_FE = ['def_X_min','def_X_max','def_X_avg','def_X_std','def_Y_min','def_Y_max','def_Y_avg','def_Y_std',
          'off_X_min','off_X_max','off_X_avg','off_X_std','off_Y_min','off_Y_max','off_Y_avg','off_Y_std']
    Plotting =['ToLeft','YardLine_std']  #'dx','dy',
    Useless_game = ['Stadium','Location','StadiumType','Turf','GameWeather','Temperature'
                             ,'Humidity','WindSpeed','WindDirection']
    Game_Id = ['HomeTeamAbbr','VisitorTeamAbbr','OffDef','TeamOnOffense']
    Game_Attri= ['PossessionTeam','DefenseTeam','NflIdRusher']
    Useless_player = ['Team','JerseyNumber','PlayerCollegeName']
    ID_feature = ['GameId','PlayId','NflId','Season'] #Game Identifiers (treat each play as not connected)
    Player_ID =['DisplayName','IsRusher'] #Player Identifiers
    cat_features = [] #Categorical features (need to come back and encode)
    for col in df.columns:
        if df[col].dtype =='object':
            cat_features.append(col)
#     print('All Cat Features: ',cat_features)
#     cat_features = [cat for cat in cat_features if cat not in Useless_game+Useless_player]
#     print('Remaining Cat: ',cat_features)
    Useless_total = Standardized+FE_out+Plotting+Player_ID+Useless_player+Game_Id+ID_feature #+Game_Attri+Useless_game #+Temp_FE
    df.drop(Useless_total,axis=1,inplace =True)
    print('Drop Cols')
    return df





### Creating CRPS score ###
def crps(y_true, y_pred):
    y_true = np.clip(np.cumsum(y_true, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    AE = ((y_true - y_pred) ** 2).sum(axis=1)
    crps_score = (((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0)) / (199 * y_true.shape[0])
    return AE,crps_score


#### Run all preprocessing and FE functions### 
def create_dataframe(df_train):
    df_train,df_rusher=cleaner(df_train)
    df_rusher = rusher_features(df_rusher)
    df_train,FE = player_features(df_train)
    df_rusher = pd.merge(df_rusher,FE, how = 'left',on='PlayId')
    dropper(df_rusher)
    return df_rusher
    
    
## Pivot player features into one row.
def player_pivot(df_train):
    
    player_columns=[]
    game_columns=[]

    for c in df_train.columns:
        if len(set(df_train[c][:22]))!= 1:
            player_columns.append(c)
        else:
            game_columns.append(c)
            
    all_columns=game_columns
    for c in player_columns:
        for i in range(22):
            all_columns.append(c+str(i))
        
    train_data=np.zeros((509762//22,len(all_columns)))
    for i in tqdm.tqdm(range(0,509762,22)):
        count=0
        for c in all_columns:
            if c in df_train:
                train_data[i//22][count] = df_train[c][i]
                count+=1
        for c in player_columns:
            for j in range(22):
                train_data[i//22][count] = df_train[c][i+j]
                count+=1
    return train_data,all_columns


#Just a dictionary to refer to back to features.
def feature_dict(key):
    feature_dict = {
        'GameId':'a unique game identifier',
        'PlayId':'a unique play identifier',
        'Team':'home or away',
        'X' : 'player position along the long axis of the field.',
        'Y' : 'player position along the short axis of the field.',
        'S' :'speed in yards/second',
        'A' : 'acceleration in yards/second^2',
        'Dis' : 'distance traveled from prior time point, in yards',
        'Orientation' : 'orientation of player (deg)',
        'Dir' : 'angle of player motion (deg)',
        'NflId': 'a unique identifier of the player',
        'DisplayName':' player\'s name',
        'JerseyNumber':'jersey number',
        'Season' : 'year of the season',
        'YardLine' : 'the yard line of the line of scrimmage',
        'Quarter' : 'game quarter (1-5, 5 == overtime)',
        'GameClock' : 'time on the game clock',
        'PossessionTeam' : 'team with possession',
        'Down' : 'the down (1-4)',
        'Distance' : 'yards needed for a first down',
        'FieldPosition' : 'which side of the field the play is happening on',
        'HomeScoreBeforePlay' : 'home team score before play started',
        'VisitorScoreBeforePlay' :'visitor team score before play started',
        'NflIdRusher' : 'the NflId of the rushing player',
        'OffenseFormation' : 'offense formation',
        'OffensePersonnel' : 'offensive team positional grouping',
        'DefendersInTheBox' : 'number of defenders lined up near the line',
        'DefensePersonnel' : 'defensive team positional grouping',
        'PlayDirection' : 'direction the play is headed',
        'TimeHandoff' :'UTC time of the handoff',
        'TimeSnap' : 'UTC time of the snap',
        'Yards' : 'the yardage gained on the play',
        'PlayerHeight' :'player height (ft-in)',
        'PlayerWeight' : 'player weight (lbs)',
        'PlayerBirthDate' : 'birth date (mm/dd/yyyy)',
        'PlayerCollegeName' : 'where the player attended college',
        'HomeTeamAbbr' : 'home team abbreviation',
        'VisitorTeamAbbr' : 'visitor team abbreviation',
        'Week' : 'week into the season',
        'Stadium' : 'stadium where the game is being played',
        'Location' : 'city where the game is being player',
        'StadiumType' : 'description of the stadium environment',
        'Turf' : 'description of the field surface',
        'GameWeather' : 'description of the game weather',
        'Temperature' : 'temperature (deg F)',
        'Humidity' : 'humidity',
        'WindSpeed' : 'wind speed in miles/hour',
        'WindDirection' : 'wind direction'
        }
    return feature_dict[key]


#Function to check any variance between the seasons
def check_var(train_df,test_df, var):
    S_2017 = train_df[var][train_df['Season'] == 2017].fillna(0)
    S_2018 = train_df[var][train_df['Season'] == 2018].fillna(0)
    S_2019 = test_df[var].fillna(0)
    
    sns.distplot(S_2017, label="2017")
    sns.distplot(S_2018, label="2018")
    sns.distplot(S_2019, label="2019")
    plt.legend(prop={'size': 12})
    plt.show()
    
    print("2017 {} mean: {:.4f}, S std: {:.4f}".format(var, S_2017.mean(), S_2017.std()))
    print("2018 {} mean: {:.4f}, S std: {:.4f}".format(var, S_2018.mean(), S_2018.std()))
    print("2019 {} mean: {:.4f}, S std: {:.4f}".format(var, S_2019.mean(), S_2019.std()))

#Function to check only 2017 to 2018 variances
def check_var(train_df, var):
    S_2017 = train_df[var][train_df['Season'] == 2017].fillna(0)
    S_2018 = train_df[var][train_df['Season'] == 2018].fillna(0)
#     S_2019 = test_df[var].fillna(0)
    
    sns.distplot(S_2017, label="2017")
    sns.distplot(S_2018, label="2018")
#     sns.distplot(S_2019, label="2019")
    plt.legend(prop={'size': 12})
    plt.show()
    
    print("2017 {} mean: {:.4f}, S std: {:.4f}".format(var, S_2017.mean(), S_2017.std()))
    print("2018 {} mean: {:.4f}, S std: {:.4f}".format(var, S_2018.mean(), S_2018.std()))
#     print("2019 {} mean: {:.4f}, S std: {:.4f}".format(var, S_2019.mean(), S_2019.std()))
    
#Another validation method utilizing the last five games to cross validate. 
def NFL_validation_split(df):
    games = df[['GameId', 'PossessionTeam']].drop_duplicates()

    # Sort so the latest games are first and label the games with cumulative counter
    games = games.sort_values(['PossessionTeam', 'GameId'], ascending=[True, False])
    games['row_number'] = games.groupby(['PossessionTeam']).cumcount() + 1

    # Use last 5 games for each team as validation. There will be overlap since two teams will have the same
    # GameId
    game_set = set([1, 2, 3, 4, 5])

    # Set of unique game ids
    game_ids = set(games[games['row_number'].isin(game_set)]['GameId'].unique().tolist())

    return game_ids

# game_ids = NFL_validation_split(train_df)

# X_train = df[~df['GameId'].isin(game_ids)]
# X_test = df[df['GameId'].isin(game_ids)]

# # Use train/test index to split target variable
# train_inds, test_inds = X_train.index, X_test.index
# y_train, y_test = y[train_inds], y[test_inds]