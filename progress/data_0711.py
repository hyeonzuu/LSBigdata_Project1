#데이터 타입
x = 15.34
print(x, "는 ", type(x), "형식입니다.", sep= '')


# 문자형 데이터 예제
a = "Hello, world!"
b = 'python programming'
print(a, type(a))
print(b, type(b))


# 여러 줄 문자열
ml_str = """This is
a multi-line
string"""
print(ml_str, type(ml_str))

#리스트
fruits = ['apple', 'banana', 'cherry']
numbers = [1, 2, 3, 4, 5]
mixed_list = [1, "Hello", [1, 2, 3]]
print("Fruits:", fruits)
print("Numbers:", numbers)
print("Mixed List:", mixed_list)

#튜플
a = (10, 20, 30, 40, 50)
a[0]
a[1] 
b = (42,)
print("좌표:", a)
print("단원소 튜플:", b)
a[3:] # 해당 인덱스 이상
a[:3] # 해당 인덱스 미만
a[1:3] # 해당 인덱스 이상 & 미만 



a = (10, 20, 30)
b = [10, 20, 30]
a[1]
b[1]
a[1] = 25
b[1] = 25

# 사용자 정의 함수
def min_max(numbers):
  return min(numbers), max(numbers)

a=[1, 2, 3, 4, 5]
result = min_max(a)
result[0] = 4
type(result)

#딕셔너리
person = {
  'name' : 'jone',
  'age' : 30,
  'city' : 'New york'
}

hyunju = {
  'name' : 'hyunju',
  'age' : (25, 24),
  'city' : ['Seoul', 'Ansung']
}

print("person: ",hyunju)
hyunju_age = hyunju.get('age')
hyunju_age[0]

# 집합
fruits = {'apple', 'banana', 'cherry', 'apple'}
print("Fruits set:", fruits) # 중복 'apple'은 제거됨

# 빈 집합 생성
empty_set = set()
print("Empty set:", empty_set)

empty_set.add("apple")
empty_set.add("apple")
empty_set.add("banana")
empty_set.remove("banana")
empty_set.discard("banana")

empty_set

# 집합 간 연산
other_fruits = {'berry', 'cherry'}
union_fruits = fruits.union(other_fruits) # 합집합
intersection_fruits = fruits.intersection(other_fruits) # 교집합
print("Union of fruits:", union_fruits)
print("Intersection of fruits:", intersection_fruits)


a=3
if (a == 2): # 불리언 값 True가 들어가면 a는 2와 같습니다. 출력, False면 a는 2와 같지 않습니다. 출력
 print("a는 2와 같습니다.")
else:
 print("a는 2와 같지 않습니다.")

# 리스트와 튜플 변환
lst = [1, 2, 3]
print("리스트:", lst)
tup = tuple(lst)
print("튜플:", tup)

set_example = {'a', 'b', 'c'}
# 딕셔너리로 변환 시, 일반적으로 집합 요소를 키 또는 값으로 사용
dict_from_set = {key: True for key in set_example}
print("Dictionary from set:", dict_from_set)

def min_max(numbers):
return min(numbers), max(numbers)

result = min_max([1, 2, 3, 4, 5])
#result = min_max((1, 2, 3, 4, 5))
print("Minimum and maximum:", result)
