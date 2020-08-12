# 感知機實現

## 資料集
可以切分為**訓練資料(training data)**與**測試資料(testing data)**
以 2:1 的比例，隨機分成訓練資料與測試資料。
Note : 注意不能把測試資料拿來訓練，否則包含測試資料訓練出來的模型又以測試資料測試的話，準確率會很高，這並不是我們要的結果。

### training data
訓練模型的資料。
    
### testing data
模型訓練完成後，testing data 帶入模型驗證可靠性。
訓練資料與測試資料來自同一個資料集，關聯性很高，

## Neural Network
單層感知機實作

## 激勵函數 activation function
使用 sign function

## HW1.py 功能
1. 設定學習率
2. 設定限制條件 (最大學習次數，避免因帶入不可二分的 dataset 而造成的程式當機)
3. 訓練資料與測試資料的比例預設為 2:1
4. 輸出
    - 鍵結值
    - 測試辨識率
5. 將訓練過程繪製成 gif 檔 (save_in_gif.py)