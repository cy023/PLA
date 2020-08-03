# 感知機實現

## 資料集
可以切分為**訓練資料(training data)**與**測試資料(testing data)**
通常會把一個資料集以 4:1 或 3:1 的比例，隨機分成訓練資料與測試資料
注意不能把測試資料拿來訓練，否則包含測試資料訓練出來的模型又以測試資料測試的話，準確率會很高，這並不是我們要的結果。
### training data
    訓練模型的資料。
    
### testing data
    模型訓練完成後，testing data 帶入模型驗證可靠性。
    訓練資料與測試資料來自同一個資料集，關聯性很高，


## 激勵函數 activation function
**利用非線性方程式，解決非線性問題**
- https://github.com/vincent732/PLA
