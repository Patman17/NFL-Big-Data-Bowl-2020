#from NFL_play_plotter import get_dx_dy, show_play_std_movement, create_football_field, PlotGame, PlotPlayer, PlotPlayId,PlotPlayIdAdv
#from NFL_play_plotter import show_play_std_movement, PlotGame, PlotPlayer, PlotPlayIdAdv
import pandas as pd
import numpy as np
pd.options.display.max_columns = 100
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import datetime
import tqdm
import random
import math


def create_football_field(linenumbers=True,
                          endzones=True,
                          highlight_line=False,
                          highlight_line_number=50,
                          highlighted_name='Line of Scrimmage',
                          down_line = False,
                          first_downline_number=0 ,
                          fifty_is_los=False,
                          figsize=(12*2, 6.33*2)):
    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.
    """
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                             edgecolor='r', facecolor='darkgreen', zorder=0,  alpha=0.5)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)

    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='white')
    if fifty_is_los:
        plt.plot([60, 60], [0, 53.3], color='gold')
        plt.text(62, 50, '<- Player Yardline at Snap', color='gold')
    # Endzones
    if endzones:
        ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='darkslategrey',
                                alpha=0.2,
                                zorder=0)
        ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='darkslategrey',
                                alpha=0.2,
                                zorder=0)
        ax.add_patch(ez1)
        ax.add_patch(ez2)
    plt.xlim(0, 120)
    plt.ylim(-5, 58.3)
    plt.axis('off')
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            plt.text(x, 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white')
            plt.text(x - 0.95, 53.3 - 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white', rotation=180)
    if endzones:
        hash_range = range(11, 110)
    else:
        hash_range = range(1, 120)

    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color='white')
        ax.plot([x, x], [53.0, 52.5], color='white')
        ax.plot([x, x], [22.91, 23.57], color='white')
        ax.plot([x, x], [29.73, 30.39], color='white')

    if highlight_line:
        hl = highlight_line_number + 10
        plt.plot([hl, hl], [0, 53.3], color='gold')
        plt.text(hl + 1, 50, '<- {}'.format(highlighted_name),
                 color='gold')
    if down_line:
        dl = first_downline_number + 10
        plt.plot([dl, dl], [0, 53.3], color='red')
        plt.text(dl + 1, 45, '<- First Down Line',
                 color='red')
    return fig, ax


# How to convert angles to dx, dy
def get_dx_dy(angle, dist):
    radian_angle = angle*math.pi/180.0
    dx = dist * math.cos(radian_angle)
    dy = dist * math.sin(radian_angle)
    return dx, dy

    
#Standardized and shows movement of all players
def show_play_std_movement(play_id, train,orient=True,objective=True,save=False):
    df = train[train.PlayId == play_id]
    YardLine = df.YardLine_std.values[0]
    Downline = YardLine + df.Distance.values[0]
    fig, ax = create_football_field(highlight_line=True,
                                    highlight_line_number=YardLine,
                                    down_line=True,
                                    first_downline_number = Downline
                                   )
    ax.scatter(df.X_std, df.Y_std, cmap='rainbow', c=~df.IsOnOffense)
    rusher_row = df[df.NflIdRusher == df.NflId]
    ax.scatter(rusher_row.X_std, rusher_row.Y_std, color='black')
    if df['Season'].values[0]!=2019:
        yards_covered = rusher_row["Yards"].values[0]
    else:
        yards_covered = 0
    
    #Direction Vector of Offense
    offense = df[df['IsOnOffense']==True]
    for (x, y, dx, dy) in zip(offense.X_std, offense.Y_std, offense.dx, offense.dy):       
#     for i in range(len(offense)):
#         x = offense['X_std'].values[i]
#         y = offense['Y_std'].values[i]
#         dx= offense['dx'].values[i]
#         dy= offense['dy'].values[i]
        ax.arrow(x, y, dx, dy, length_includes_head=True, width=0.08, color='yellow')
        
    #Direction Vector of Defense  
    defense = df[df['IsOnOffense']==False]
    for (x, y, dx, dy) in zip(defense.X_std, defense.Y_std, defense.dx, defense.dy):
#     for i in range(len(defense)):
#         x = defense['X_std'].values[i]
#         y = defense['Y_std'].values[i]
#         dx= defense['dx'].values[i]
#         dy= defense['dy'].values[i]
        ax.arrow(x, y, dx, dy, length_includes_head=True, width=0.08, color='pink')
    #Rusher Plot   
    x = rusher_row["X_std"].values[0]
    y = rusher_row["Y_std"].values[0]
    rusher_dir = rusher_row["Dir_std"].values[0]
    rusher_speed = rusher_row["S"].values[0]
    dx, dy = get_dx_dy(rusher_dir, rusher_speed)
    ax.arrow(x, y, dx, dy, length_includes_head=True, width=0.1, color='black')
    
    if df['Season'].values[0]!=2019 and objective==True:
        New_YL = YardLine + yards_covered +10
        plt.plot([New_YL, New_YL], [0, 53.3], color='darkblue')
        plt.text(New_YL + 1, 2, '<- New LOS', color='darkblue')    
        
################# Orientation Vectors ####################      
    if orient == True:
        #Orientation Vector of Offense
        size = 0.55
        off_orient_dx=[]
        off_orient_dy=[]
        for (x, y, Orient) in zip(offense.X_std, offense.Y_std, offense.Orientation_std):
            dx, dy = get_dx_dy(Orient,size)
#         for i in range(len(offense)):
#             orient = offense['Orientation_std'].values[i]
#             orient_dx, orient_dy = get_dx_dy(orient, size)
#             off_orient_dx.append(orient_dx)
#             off_orient_dy.append(orient_dy)
#         for i in range(len(offense)):
#             x = offense['X_std'].values[i]
#             y = offense['Y_std'].values[i]
#             dx= off_orient_dx[i]
#             dy= off_orient_dy[i]
            ax.arrow(x, y, dx, dy, length_includes_head=False, head_width = 0.001, head_length=0.001, width=0.07, color='lime')
            
            
        #Orientation Vector of Defense
        def_orient_dx=[]
        def_orient_dy=[]
        for (x, y, Orient) in zip(defense.X_std, defense.Y_std, defense.Orientation_std):
            dx, dy = get_dx_dy(Orient,size)
#         for i in range(len(defense)):
#             orient = defense['Orientation_std'].values[i]
#             orient_dx, orient_dy = get_dx_dy(orient, size)
#             def_orient_dx.append(orient_dx)
#             def_orient_dy.append(orient_dy)
#         for i in range(len(defense)):
#             x = defense['X_std'].values[i]
#             y = defense['Y_std'].values[i]
#             dx= def_orient_dx[i]
#             dy= def_orient_dy[i]
            ax.arrow(x, y, dx, dy, length_includes_head=False, head_width = 0.001, head_length=0.001, width=0.07, color='lime')  
    
################## Plotting Labels ######################
    Week  = df['Week'].values[0]
    Season= df['Season'].values[0]
    gameid = df['GameId'].values[0]
    OffTeam = df['PossessionTeam'].values[0]
    DefTeam = df['DefenseTeam'].values[0]
    Name = rusher_row['DisplayName'].values[0]
    Position = rusher_row['Position'].values[0]
    S = rusher_row['S'].values[0]
    A = rusher_row['A'].values[0]
    Quarter = df['Quarter'].values[0]
    GameClock = df['GameClock'].values[0]
    Team = rusher_row['Team'].values[0].title()
    HomeScore = df['HomeScoreBeforePlay'].values[0]
    VisitScore = df['VisitorScoreBeforePlay'].values[0]
    DisDown = df['Distance'].values[0]
    Down = df['Down'].values[0]
    
    ## Extra stuff
    Orient = np.round(rusher_row['Orientation_std'].values[0],2)
    Dir = np.round(rusher_row['Dir_std'].values[0],2)

    plt.title(f'Week: {Week}, Season: {Season}, Offense: {OffTeam}, Defense: {DefTeam}  \n'
              f'Play #: {play_id}, Game #: {gameid}  \n' 
              f'Runner: {Name}, Position: {Position}, Speed: {S} yd/s , Acceleration {A} yd/s^2\n' 
              f'Down: {Down}, Yd to 1st Down: {DisDown}, LOS = {YardLine} yd, Quarter: {Quarter}, Game Clock: {GameClock} sec\n'
              f'Team: {Team}, Home Score: {HomeScore} Pts, Away Score: {VisitScore} Pts \n'
              f'Yards Gain = {yards_covered} yd \n'
              f'Direction = {Dir}°, Orientation = {Orient}°'
              ,fontsize=20)
#     plt.legend()
#     plt.show()
    if save == True:
        plt.savefig(f'{Name}_{play_id}.png')
    else:
        plt.show()
    
    
#Plot Whole/Team Game Rush Plays
def PlotGame(df_train,whole_game = True,TeamName ='NE', Season = 2017, Week = 1,orient=True):
    bye = 0
    if whole_game == True:
        WeekGame=df_train[(df_train.Season == Season)& (df_train.Week ==Week)]
        if np.in1d(TeamName,WeekGame.PossessionTeam.unique())[0] == False:
            print('No game found.')
            bye=1
        else:
            GameId =WeekGame[WeekGame['PossessionTeam']==TeamName]['GameId'].values[0]
            TeamGame=WeekGame[WeekGame['GameId']==GameId]
    elif whole_game ==False:
        WeekGame = df_train[(df_train.Season == Season)& (df_train.Week ==Week)]
        if np.in1d(TeamName,WeekGame.PossessionTeam.unique())[0] == False:
            print('No game found.')
            bye=1
        else:
            TeamGame = df_train[(df_train.PossessionTeam == TeamName)&(df_train.Season == Season)& (df_train.Week ==Week)]
    if bye==0:       
        for i,play in enumerate(TeamGame['PlayId'].unique()):
            print('Play #:',i+1)
            show_play_std_movement(play,df_train,orient)
    elif bye ==1:
        print('Try Again.')
    return


#Plot Individual Players Rush Plays Per Game
def PlotPlayer(df_train,Name ='Carlos Hyde', Season = 2017, Week = 1,orient=True):
    NflId= df_train[df_train.DisplayName==Name]['NflId'].values[0]
    PlayerGame = df_train[(df_train.DisplayName==Name)&(df_train.Season==Season)&(df_train.Week==Week)&(df_train.NflIdRusher==NflId)]
    for i,play in enumerate(PlayerGame['PlayId'].unique()):
        print('Play #:',i+1)
        show_play_std_movement(play,df_train,orient)
    return


#Plot Per PlayID
def PlotPlayId(df_train,PlayId=[],orient=True,objective_=True,save_=True):
    for i,play in enumerate(PlayId):
        print('Play #:',i+1)
        show_play_std_movement(play,df_train,orient=True,objective=objective_,save=save_)
    return


#Plot on Criteria Per PlayID
def PlotPlayIdAdv(df_train,Search_df,orient=True):
    #Takes in a dataframe that has your playids in column 0 and attribute of interest in column 1
    #TimeDelta = df_train[df_train['TimeDelta']>2][['PlayId','TimeDelta']].drop_duplicates().sort_values('TimeDelta')
    #LongYards = df_train[df_train['Yards']>90][['PlayId','Yards']].drop_duplicates().sort_values('Yards')
    PlayId=Search_df.iloc[:,0].values
    Attribute_title =Search_df.columns[1]
    Attributes = Search_df.iloc[:,1].values
    for i,play in enumerate(PlayId):
        Attri = Attributes[i]
        print('Play #:',i+1)
        print(f'{Attribute_title}: {Attri}')
        show_play_std_movement(play,df_train,orient)
    return


############# EXAMPLE USAGE ###########

# Sample Plotting Whole Game
# PlotGame(df_train,whole_game = False,TeamName ='ARZ', Season = 2017, Week = 11,orient=True)

# Plotting Player
# PlotPlayer(df_train,Name ='Aaron Jones', Season = 2018, Week = 4,orient=True)

# PlotPlayId(df_train,PlayId=[],orient=True)

# LongYards = df_train[df_train['Yards']>30][['PlayId','Yards']].drop_duplicates().sort_values('Yards')
# PlotPlayIdAdv(df_train,LongYards,orient=True)

# Ace = df_train[(df_train['OffenseFormation']=='ACE')|(df_train['OffenseFormation']=='EMPTY')|(df_train['OffenseFormation']=='WILDCAT')][['PlayId','OffenseFormation']].drop_duplicates().sort_values('OffenseFormation')
# PlotPlayIdAdv(df_train,Ace,orient=True)













def cleaner2(df_train,misc=True):
    print('starting.....')
    #abbreviations
    df_train.loc[df_train.VisitorTeamAbbr == "ARI", 'VisitorTeamAbbr'] = "ARZ"
    df_train.loc[df_train.HomeTeamAbbr == "ARI", 'HomeTeamAbbr'] = "ARZ"

    df_train.loc[df_train.VisitorTeamAbbr == "BAL", 'VisitorTeamAbbr'] = "BLT"
    df_train.loc[df_train.HomeTeamAbbr == "BAL", 'HomeTeamAbbr'] = "BLT"

    df_train.loc[df_train.VisitorTeamAbbr == "CLE", 'VisitorTeamAbbr'] = "CLV"
    df_train.loc[df_train.HomeTeamAbbr == "CLE", 'HomeTeamAbbr'] = "CLV"

    df_train.loc[df_train.VisitorTeamAbbr == "HOU", 'VisitorTeamAbbr'] = "HST"
    df_train.loc[df_train.HomeTeamAbbr == "HOU", 'HomeTeamAbbr'] = "HST"
    print('abb')
    #important identifiers
    df_train['ToLeft'] = df_train.PlayDirection == "left"
    df_train['IsRusher'] = df_train.NflId == df_train.NflIdRusher   
    df_train['OffDef'] = 'Defense'
    df_train.loc[((df_train.Team == 'home')&(df_train.PossessionTeam==df_train.HomeTeamAbbr)) | ((df_train.Team == 'away') & (df_train.PossessionTeam == df_train.VisitorTeamAbbr)),'OffDef'] = 'Offense'
    df_train.loc[df_train['NflId']==df_train['NflIdRusher'],'OffDef'] ='Rusher'
    print('Identifiers')
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
    print('Std Orientation')
    #Correction of 2017 Orientation being 90 off.
    df_train.loc[df_train['Season'] == 2017, 'Orientation_std'] = np.mod(-90 + df_train.loc[df_train['Season'] == 2017, 'Orientation_std'], 360)
    #Direction Vector
    df_train['dx'] = df_train.S*np.cos((df_train.Dir_std)*np.pi/180.0)
    df_train['dy'] = df_train.S*np.sin((df_train.Dir_std)*np.pi/180.0)
    df_train['A_dx'] = df_train.A*np.cos((df_train.Dir_std)*np.pi/180.0)
    df_train['A_dy'] = df_train.A*np.sin((df_train.Dir_std)*np.pi/180.0)
    print('Direction Vector')
    #Standardizing Speed (2017 to everything else)
#     traindf.loc[traindf['Season'] == 2017, 'S'] = (traindf['S'][traindf['Season'] == 2017] - 2.4355) / 1.2930 * 1.4551 + 2.7570
    #PlayerHeight
    df_train['PlayerHeight'] = df_train['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))
    print('PlayerHeight')


    #Attribute Change
#     df_train['Down']=df_train['Down'].astype('object')
#     df_train['Quarter']=df_train['Quarter'].astype('object')
#     df_train['YardLine_std_ob']=df_train['YardLine_std'].astype('object')
    print('Dtype Change')
    #Scoring and conversion of Home/Away to Actual Team Names
    df_train.loc[df_train['PossessionTeam']==df_train['VisitorTeamAbbr'],'OffenseHome'] = 0
    df_train.loc[df_train['PossessionTeam']==df_train['HomeTeamAbbr'],'OffenseHome'] = 1
    df_train.loc[df_train['OffenseHome']==1,'DefenseTeam']=df_train['VisitorTeamAbbr']
    df_train.loc[df_train['OffenseHome']==0,'DefenseTeam']=df_train['HomeTeamAbbr']
    df_train.loc[df_train['OffenseHome']==1,'OffScoreDiff']=df_train['HomeScoreBeforePlay']-df_train['VisitorScoreBeforePlay']
    df_train.loc[df_train['OffenseHome']==0,'OffScoreDiff']=df_train['VisitorScoreBeforePlay']-df_train['HomeScoreBeforePlay']
    print('Scoring')
    #GameClock in Seconds
    def strtoseconds(txt):
        txt = txt.split(':')
        ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60
        return ans
    df_train['GameClock'] = df_train['GameClock'].apply(strtoseconds)
    #Time Elasped in Game in Minutes
    df_train.loc[df_train['Quarter']< 5,'TimeElapsed'] = ((df_train.loc[df_train['Quarter']< 5,'Quarter']-1.0)*900.0+(900.0-df_train.loc[df_train['Quarter']< 5,'GameClock']))/60.0
    df_train.loc[df_train['Quarter']==5,'TimeElapsed'] = ((df_train.loc[df_train['Quarter']==5,'Quarter']-1.0)*900.0+(600.0-df_train.loc[df_train['Quarter']==5,'GameClock']))/60.0
    print('GameClock & Time Elasped')
    #Conversion to DateTime
    print('Start Conversion to DateTime')
    df_train['TimeHandoff'] = df_train['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    df_train['TimeSnap'] = df_train['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    df_train['PlayerBirthDate'] = df_train['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))
    print('Converted to DateTime')
    #Time Features
    seconds_in_year = 60*60*24*365.25
    df_train['TimeDelta'] = (df_train['TimeHandoff'] - df_train['TimeSnap']).map(lambda x:x.seconds)
    df_train['TimeDelta'] = df_train.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
    df_train['PlayerAge'] = df_train.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)
    # df_train['Year'] = df_train.apply(lambda row: row['TimeHandoff'].year, axis =1) #similar to season
    #df_train['Month'] = df_train.apply(lambda row: row['TimeHandoff'].month,axis=1)
    #df_train['Hour'] = df_train.apply(lambda row: row['TimeHandoff'].hour,axis =1)
    print('Time Features')
    #Rusher Position Classification
#     def position_indicator(x):
#         if x == 'RB' or x == 'HB':
#             return 'RB'
#         elif x =='QB':
#             return 'QB'
#         else:
#             return 'Trick'
#     df_train['Position']=df_train['Position'].apply(position_indicator)
#     #Offense Formation Classification
#     def Offense_formation(x):
#         if x == 'SINGLEBACK':
#             return 'SINGLEBACK'
#         elif x =='SHOTGUN':
#             return 'SHOTGUN'
#         elif x =='I_FORM':
#             return 'I_FORM'
#         elif x == 'PISTOL':
#             return 'PISTOL'
#         elif x == 'JUMBO':
#             return 'JUMBO'
#         elif x == 'WILDCAT':
#             return 'WILDCAT'
#         else:
#             return 'OTHER'
#     df_train['OffenseFormation']=df_train['OffenseFormation'].apply(Offense_formation)
    
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
    
    df_train['DefensePersonnel'] = df_train['DefensePersonnel'].apply(lambda x: split_personnel(x))
    df_train['DefensePersonnel'] = df_train['DefensePersonnel'].apply(lambda x: defense_formation(x))
    df_train['num_DL'] = df_train['DefensePersonnel'].apply(lambda x: x[0])
    df_train['num_LB'] = df_train['DefensePersonnel'].apply(lambda x: x[1])
    df_train['num_DB'] = df_train['DefensePersonnel'].apply(lambda x: x[2])
    
    df_train['OffensePersonnel'] = df_train['OffensePersonnel'].apply(lambda x: split_personnel(x))
    df_train['OffensePersonnel'] = df_train['OffensePersonnel'].apply(lambda x: offense_formation(x))
    df_train['num_QB'] = df_train['OffensePersonnel'].apply(lambda x: x[0])
    df_train['num_RB'] = df_train['OffensePersonnel'].apply(lambda x: x[1])
    df_train['num_WR'] = df_train['OffensePersonnel'].apply(lambda x: x[2])
    df_train['num_TE'] = df_train['OffensePersonnel'].apply(lambda x: x[3])
    df_train['num_OL'] = df_train['OffensePersonnel'].apply(lambda x: x[4])
    df_train.drop(['OffensePersonnel','DefensePersonnel'], axis=1, inplace=True)
    # Let's create some features to specify if the OL is covered
#     df_train['OL_diff'] = df_train['num_OL'] - df_train['num_DL']
#     df_train['OL_TE_diff'] = (df_train['num_OL'] + df_train['num_TE']) - df_train['num_DL']
#     # Let's create a feature to specify if the defense is preventing the run
#     # Let's just assume 7 or more DL and LB is run prevention
#     df_train['run_def'] = (df_train['num_DL'] + df_train['num_LB'] > 6).astype(int)
#     df_train.drop(['OffensePersonnel','DefensePersonnel'], axis=1, inplace=True)
    print('Personnel')
    
#     #Offense Personnel --> Offensive Personnel Counter
#     def process_offense(string):
#         str_ = string.split(',')
#         rb=0
#         te=0
#         wr=0
#         ol=0
#         dl=0
#         lb=0
#         db=0
#         qb=0
#         for s in str_:
#             if 'RB' in s:
#                 rb=int(s.split()[0])
#             if 'TE' in s:
#                 te=int(s.split()[0])
#             if 'WR' in s:
#                 wr=int(s.split()[0])
#             if 'OL' in s:
#                 ol=int(s.split()[0])
#             if 'DL' in s:
#                 dl=int(s.split()[0])
#             if 'LB' in s:
#                 lb=int(s.split()[0])
#             if 'DB' in s:
#                 db=int(s.split()[0])
#             if 'QB' in s:
#                 qb=int(s.split()[0])
#         return [rb,te,wr,ol,dl,lb,db,qb]

#     offense_counts=df_train['OffensePersonnel'].apply(process_offense)
#     r,t,w,o,d,l,b,q=list(map(list,zip(*offense_counts)))

#     df_train['Off_RB']=r
#     df_train['Off_TE']=t
#     df_train['Off_WR']=w
#     df_train['Off_OL']=o
#     df_train['Off_DL']=d
#     df_train['Off_LB']=l
#     df_train['Off_DB']=b
#     df_train['Off_QB']=q
#     print('Off Personnel')
    
#     #Defense Personnel --> Defensive Personnel Counter
#     def process_defense(x):
#         num=[]
#         num=x.split(',')
#         dl=int(num[0].split(' ')[0])
#         lb=int(num[1].split(' ')[1])
#         db=int(num[2].split(' ')[1])
#         if(len(num)>3):
#              ol=int(num[3].split(' ')[1])
#         else:
#              ol=0
#         return [dl,lb,db,ol]

#     defense_counts=df_train['DefensePersonnel'].apply(process_defense)
#     u,v,x,y=list(map(list,zip(*defense_counts)))

#     df_train['Def_DL']=u
#     df_train['Def_LB']=v
#     df_train['Def_BL']=x
#     df_train['Def_OL']=y
#     print('Def Personnel')
    
    
############################################### MISC Cleaning Columns ###############################################
    if misc:
        #Game Weather
        df_train['GameWeather'] = df_train['GameWeather'].str.lower()
        indoor = "indoor"
        df_train['GameWeather'] = df_train['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)
        df_train['GameWeather'] = df_train['GameWeather'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)
        df_train['GameWeather'] = df_train['GameWeather'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
        df_train['GameWeather'] = df_train['GameWeather'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
        
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
        df_train['GameWeather'] = df_train['GameWeather'].apply(give_me_GameWeather)
        print('GameWeather')
        
        
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
        df_train['WindSpeed'] = df_train['WindSpeed'].apply(lambda x: str(x).lower().replace('mph', '').strip() if not pd.isna(x) else x) # strip ones with mph
        df_train['WindSpeed'] = df_train['WindSpeed'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x) # had some x-y so take the avg
        df_train['WindSpeed'] = df_train['WindSpeed'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x) # had one with gust up to. avg it
        
        def str_to_float(txt):
            try:
                return float(txt)
            except:
                return -1

        df_train['WindSpeed'] = df_train['WindSpeed'].apply(str_to_float)
        print('Wind Speed')
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
        
        df_train['WindDirection'] = df_train['WindDirection'].replace(north,'north')
        df_train['WindDirection'] = df_train['WindDirection'].replace(south,'south')
        df_train['WindDirection'] = df_train['WindDirection'].replace(west,'west')
        df_train['WindDirection'] = df_train['WindDirection'].replace(east,'east')
        df_train['WindDirection'] = df_train['WindDirection'].replace(north_east,'north_east')
        df_train['WindDirection'] = df_train['WindDirection'].replace(north_west,'north_west')
        df_train['WindDirection'] = df_train['WindDirection'].replace(south_east,'clear')
        df_train['WindDirection'] = df_train['WindDirection'].replace(south_west,'south_west')
        df_train['WindDirection'] = df_train['WindDirection'].replace(no_wind,'no_wind')
        df_train['WindDirection'] = df_train['WindDirection'].replace(nan,np.nan)
        print('Wind Direction')
        #Stadium Type clean up
        outdoor = ['Outdoor', 'Outdoors', 'Cloudy', 'Heinz Field', 'Outdor', 'Ourdoor', 
                   'Outside', 'Outddors','Outdoor Retr Roof-Open', 'Oudoor', 'Bowl']

        indoor_closed = ['Indoors', 'Indoor', 'Indoor, Roof Closed', 'Indoor, Roof Closed', 'Retractable Roof',
                         'Retr. Roof-Closed', 'Retr. Roof - Closed', 'Retr. Roof Closed']

        indoor_open   = ['Indoor, Open Roof', 'Open', 'Retr. Roof-Open', 'Retr. Roof - Open']
        dome_closed   = ['Dome', 'Domed, closed', 'Closed Dome', 'Domed', 'Dome, closed']
        dome_open     = ['Domed, Open', 'Domed, open']

        df_train['StadiumType'] = df_train['StadiumType'].replace(outdoor,'outdoor')
        df_train['StadiumType'] = df_train['StadiumType'].replace(indoor_closed,'indoor_closed')
        df_train['StadiumType'] = df_train['StadiumType'].replace(indoor_open,'indoor_open')
        df_train['StadiumType'] = df_train['StadiumType'].replace(dome_closed,'dome_closed')
        df_train['StadiumType'] = df_train['StadiumType'].replace(dome_open,'dome_open')
        print('Stadium Type')
        # Turf Clean up
        natural_grass = ['natural grass','Naturall Grass','Natural Grass']
        grass = ['Grass']

        fieldturf = ['FieldTurf','Field turf','FieldTurf360','Field Turf']

        artificial = ['Artificial','Artifical']

        df_train['Turf'] = df_train['Turf'].replace(natural_grass,'natural_grass')
        df_train['Turf'] = df_train['Turf'].replace(grass,'grass')
        df_train['Turf'] = df_train['Turf'].replace(fieldturf,'fieldturf')
        df_train['Turf'] = df_train['Turf'].replace(artificial,'artificial')
        print('Turf')
        #Temperature and Humidity FFill
        df_train['Humidity'].fillna(method='ffill', inplace=True)
        df_train['Temperature'].fillna(method='ffill', inplace=True)
        print('Temp & Humidity')
    
    
    return df_train