from sklearn import tree

# b1: Thu thập dữ liệu
# b2: Xử lý dữ liệu
# b3: Xây dựng model
# b4: Dự đoán kết quả
# b5: Đánh giá xem model có hiệu quả không?

decisionTree = tree.DecisionTreeClassifier()

feature = [[1, 3, 3, 7],
           [5, 2, 4, 6],
           [1, 2, 4, 6],
           [5, 4, 4, 3],
           [1, 4, 4, 7],
           [3, 2, 3, 7],
           [3, 3, 3, 6],
           [5, 2, 2, 7]
           ]

label = [0, 1, 1, 0, 0, 0, 0, 1]

result = decisionTree.fit(feature, label)

final = result.predict([[1, 4, 3, 6]])

print(final)
