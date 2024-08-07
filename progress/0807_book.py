import json
import pandas as pd
import numpy as np


geo_seoul = json.load(open('bigfile/SIG_Seoul.geojson', encoding = 'UTF-8'))

type(geo_seoul)
len(geo_seoul)

geo_seoul.keys()
geo_seoul["features"][0]
len(geo_seoul["features"])
type(geo_seoul["features"]) 
geo_seoul["features"][0].keys()
geo_seoul["features"][0]["properties"]
geo_seoul["features"][0]["geometry"]
coordinates_list =  geo_seoul["features"][0]["geometry"]["coordinates"]
len(coordinates_list) # 1 , 대괄호 4개
len(coordinates_list[0])  # 1, 대괄호 3개
len(coordinates_list[0][0]) # 2332개, 이제 df로 만들기


coordinates_array = np.array(coordinates_list[0][0])
x = coordinates_array[:,0]
y = coordinates_array[:,1]


import matplotlib.pyplot as plt
#plt.plot(x[::10], y[::10]) # 10개에 1개씩
plt.plot(x, y)
plt.show()
plt.clf()

geo_seoul["features"][0]["properties"] # 종로구
geo_seoul["features"][1]["properties"] # 중구

# 함수로 만들기
def draw_seoul(num):
    gu_name = geo_seoul["features"][num]["properties"]['SIG_KOR_NM']
    coordinates_list=geo_seoul["features"][num]["geometry"]["coordinates"]
    coordinate_array=np.array(coordinate_list[0][0])
    x = coordinate_array[:,0]
    y = coordinate_array[:,1]
    
    plt.rcParams.update({"font.family" : "Malgun Gothic"}) # 출력되기전에 한글 설정 해야함 
    plt.plot(x, y)
    plt.title(gu_name)
    plt.show()
    plt.clf()
    
draw_seoul(10)
===============================================================================
# 서울 전체 지도
# df 생성
def gu_df(num):
    gu_name = geo_seoul["features"][num]['properties']["SIG_KOR_NM"]  # 각 구의 이름 추출
    coordinate_list = geo_seoul["features"][num]["geometry"]["coordinates"]  # 각 구의 좌표 리스트 추출

    
    # 각 폴리곤의 좌표 배열로 변환 후 결합
    coordinates = np.concatenate([np.array(i) for i in coordinate_list])
    
    # 데이터프레임 생성
    df = pd.DataFrame({
        'gu_name': [gu_name] * len(coordinates),  # 각 좌표에 구 이름 할당
        'x': coordinates[:, 0],  # x 좌표 (경도)
        'y': coordinates[:, 1]  # y 좌표 (위도)
    })
    
    return df

def draw_seoul(num):
    gu_name = df["gu_name"]
    x = df["x"]
    y = df["y"]
    
    plt.rcParams.update({"font.family" : "Malgun Gothic"}) # 출력되기전에 한글 설정 해야함 
    plt.plot(x, y)
    plt.title(gu_name)
    plt.show()
    plt.clf()
    
-------------------------------------------------------------------------------

# #1
# def draw_seoul_all(geo_seoul):
#     plt.rcParams.update({"font.family" : "Malgun Gothic"})  # 한글 폰트 설정
# 
#     for feature in geo_seoul["features"]:
#         gu_name = feature["properties"]['SIG_KOR_NM']
#         coordinate_list = feature["geometry"]["coordinates"]
#         
#         # 폴리곤 데이터가 여러 개 있을 경우를 대비하여 루프를 돌림
#         for polygon in coordinate_list:
#             coordinate_array = np.array(polygon[0])
#             x = coordinate_array[:, 0]
#             y = coordinate_array[:, 1]
#             plt.plot(x, y, label=gu_name)
#     
#     plt.title('서울시 전체 지도')
#     plt.xlabel('경도')
#     plt.ylabel('위도')
#     plt.legend(loc='upper right', fontsize='small')
#     plt.show()
#     plt.clf()
# 
# # 가정된 geo_seoul 변수를 전달하여 함수 호출
# draw_seoul_all(geo_seoul)
# 
# #2
# geo_mex=[]
# geo_mey=[]
# geo_name=[]
# for x in np.arange(0,25):
#     gu_name=geo_seoul["features"][x]["properties"]['SIG_KOR_NM']
#     coordinates_list=geo_seoul["features"][x]["geometry"]["coordinates"]
#     coordinate_array=np.array(coordinates_list[0][0])
#     geo_mex.append(coordinate_array[:,0])
#     geo_mey.append(coordinate_array[:,1])
#     geo_name.append(geo["features"][x]["properties"]['SIG_KOR_NM'])
# for x in np.arange(0,25):
#     plt.plot(geo_mex[x],geo_mey[x])
#     plt.show()
#     
# plt.clf() 
# 
# # 3
# gu_name = [geo_seoul["features"][x]["properties"]["SIG_KOR_NM"] for x in range(len(geo_seoul["features"]))]
# gu_name
# 
# coordinate_list = [geo_seoul["features"][x]["geometry"]['coordinates'] for x in range(len(geo_seoul["features"]))]
# coordinate_list
# 
# np.array(coordinate_list[0][0][0])
# 
# import numpy as np
# import pandas as pd
# 
# # 남규님 code - 종로구
# pd.DataFrame({'gu_name' : [gu_name[0]] * len(np.array(coordinate_list[0][0][0])),
#               'x'       : np.array(coordinate_list[0][0][0])[:,0],
#               'y'       : np.array(coordinate_list[0][0][0])[:,1]})
#               
# # 한결 생각 - 얘는 왜인지 모르겠으나 syntax error 발생
# # pd.DataFrame({'gu_name' : [gu_name[x]] * len(np.array(coordinate_list[x][0][0]) for x in range(len(geo_seoul["features"]))],
# #               'x'       : [np.array(coordinate_list[x][0][0])[:,0] for x in range(len(geo_seoul["features"]))],
# #               'y'       : [np.array(coordinate_list[x][0][0])[:,1]] for x in range(len(geo_seoul["features"])) })
# 
# # 빈 리스트 생성
# empty = []
# 
# # for in 구문을 이용하하여 geo_seoul["features"]의 길이만큼 for 문 안의 내용을 반복
# for x in range(len(geo_seoul["features"])):
#     df = pd.DataFrame({
#         'gu_name': [gu_name[x]] * len(np.array(coordinate_list[x][0][0])),
#         'x': np.array(coordinate_list[x][0][0])[:, 0],
#         'y': np.array(coordinate_list[x][0][0])[:, 1]
#     })
#     empty.append(df)
# 
# # 모든 DataFrame을 하나로 합치기, ignore_index=True를 이용하여 기존의 인덱스를 무시하고 새로운 인덱스 부여
# seoul_total = pd.concat(empty, ignore_index=True)
# seoul_total
# 
# import seaborn as sns
# sns.scatterplot(data = seoul_total, x='x', y='y', hue="gu_name", s=5)
# # plt.plot(x,y, hue="gu_name")
# plt.show()
# plt.clf()

----------------------
# 4 선생님과 함꼐
# 구 이름 만들기
gu_name = list()

for i in range(25):
    gu_name.append(geo_seoul['features'][i]['properties']['SIG_KOR_NM'])
    
gu_name

# x, y 판다스 데이터 프레임 만들기
def make_seouldf(num):
    gu_name = geo_seoul['features'][num]['properties']['SIG_KOR_NM']
    coordinate_list = geo_seoul['features'][num]['geometry']['coordinates']
    coordinate_list = np.array(coordinate_list[0][0])
    x = coordinate_list[:, 0]
    y = coordinate_list[:, 1]
    
    return pd.DataFrame({"gu_name":gu_name, "x": x, "y": y})

make_seouldf(1)
# 데이터프레임에 넣기
result=pd.DataFrame({})
for i in range(25):
    result=pd.concat([result, draw_seoul(i)], ignore_index=True)    

result[["x"]]



# 서울 그래프 그리기
import seaborn as sns
sns.scatterplot(data= result, x="x", y = "y", hue = "gu_name",
                palette="deep", s=2, legend=False)
#result.plot(kind = "scatter",x="x", y = "y") # y를 선으로 잇는다
plt.show()
plt.clf()

# 강남만 다른 색으로 표현
gangnam_df = result.assign(is_gangnam = np.where(result["gu_name"]=="강남구", "강남", "안강남"))
sns.scatterplot(data= gangnam_df, x="x", y = "y", hue = "is_gangnam", s=2,
                legend=False,  palette=["gray", "red"])
plt.show()
plt.clf()

--------------------------------------------------------------------------------
교재 310p

geo_seoul = json.load(open('bigfile/SIG_Seoul.geojson', encoding = 'UTF-8'))
df_pop = pd.read_csv("data/Population_SIG.csv")
df_seoulpop = df_pop.iloc[1:26]
df_seoulpop["code"] = df_seoulpop["code"].astype(str)
df_seoulpop.info()

import folium
# center_x = result["x"].mean() # 126.97315486480478
# center_y = result["y"].mean() # 37.55180997129064

map_sig = folium.Map(location = [37.551, 126.973], # 지도 중심좌표 찍어줘야함
            zoom_start = 11,  # 확대 단계
            tiles = "CartoDBpositron ")            

# map_sig.save("map_seoul.html")

# 구역 나누기 : 코로플릿
folium.Choropleth(
    geo_data = geo_seoul, # 지도 데이터
    data = df_seoulpop,   # 통계 데이터
    columns = ("code", "pop"), # 행정구역 코드, 인구
    key_on = "feature.properties.SIG_CD").add_to(map_sig) # geo 행정 구역 코드

# map_sig.save("map_seoul.html")

# 계급구간 정하기
bins = df_seoulpop["pop"].quantile([0, 0.2, 0.4, 0.6 ,0.8, 1])

# 디자인 수정하기
map_sig = folium.Choropleth(
    geo_data = geo_seoul, # 지도 데이터
    data = df_seoulpop,   # 통계 데이터
    columns = ("code", "pop"), # 행정구역 코드, 인구
    key_on = "feature.properties.SIG_CD",
    fill_color = "YlGnBu", # 컬러맵
    fill_opacity = 1, # 투명도
    line_opacity = 0.5, # 경계선 투명도
    bins = bins).add_to(map_sig) # 계급구간 기준값
    
map_sig.save("map_seoul.html")

# 점 찍는 법
# make_seouldf(0).iloc[:, 1:2].mean()
# make_seouldf(10).iloc[:, 1:2].mean()
# make_seouldf(10).iloc[:, 2:3].mean()

folium.Marker([37.583744, 126.983800], popup="종로구").add_to(map_sig)
folium.Marker([37.641143, 127.06859], popup="노원구").add_to(map_sig)
map_sig.save("map_seoul.html")





