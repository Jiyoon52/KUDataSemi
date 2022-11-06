from KETIPrePartialDataPreprocessing.error_detection import dataOutlier
import numpy as np
class unCertainErrorRemove():
    '''Let UnCertain Outlier from DataFrame Data to NaN. This function makes more Nan according to the data status.
    
    **Data Preprocessing Modules**::

            neighbor_error_detected_data
    '''
    def __init__(self, data, param):
  
        self.param = param
        data_outlier = dataOutlier.DataOutlier(data)
        self.data = data_outlier.refinmentForOutlierDetection()
    
    def getNoiseIndex(self):
        """    
        :return result: Noise Index
        :type: json

        self.outlierIndex
        self.mergedOutlierIndex

        """
        outlierDetectorConfigs =self.param['outlierDetectorConfig']
        MLList = [ 'IF', 'KDE', 'LOF', 'MoG', 'SR']
        self.outlierIndex={}
        for outlierDetectorConfig in outlierDetectorConfigs:
            algorithm = outlierDetectorConfig['algorithm']
            if algorithm in MLList:
                IndexResult = self.getOutlierIndexByMLOutlierDetector(outlierDetectorConfig)
            elif algorithm == "IQR" :
                IndexResult  = self.getOutlierIndexByIQR(outlierDetectorConfig)
            elif algorithm == "SD":
                IndexResult  = self.getOutlierIndexBySeasonalDecomposition(outlierDetectorConfig['alg_parameter'])
            self.outlierIndex[algorithm] = IndexResult 
        
        self.mergedOutlierIndex={}
        for column in self.data.columns:
            self.mergedOutlierIndex[column]= []
            for outlierDetectorConfig in outlierDetectorConfigs:
                algorithm = outlierDetectorConfig['algorithm']
                self.mergedOutlierIndex[column].extend(self.outlierIndex[algorithm][column])
        
        return self.mergedOutlierIndex

    def getIntersectionIndex(self, outlierIndex):
        """    
        :param outlierIndex: Noise Index
        :type outlierIndex: json

        :return intersectionIndex: Intersection index by each noise index key
        :type: list
        """
        first_key= list(outlierIndex.keys())[0]
        intersectionIndex = outlierIndex[first_key]
        for key, value in outlierIndex.items():
            intersectionIndex = list(set(intersectionIndex) & set(list(value)))
        return intersectionIndex

    def getOutlierIndexByIQR(self, param):
        """    
        :param param: having 'weight' parameter. weight is IQR duration adjustment parameter.
        :type weight: json

        :return outlier_index: Intersection index by each noise index key
        :type: list
        """
        df = self.data.copy()
        weight = param['alg_parameter']['weight']
        outlier_index={}
        for column in df.columns:
            column_x = df[column]
            # 1/4 분위와 3/4 분위 지점을 np.percentile로 구함
            quantile_25 = np.nanpercentile(column_x.values, 25)
            quantile_75 = np.nanpercentile(column_x.values, 75)
            print("===== ",column, "- 25%:", quantile_25, "75%", quantile_75)

            # IQR을 구하고 IQR에 1.5를 곱해 최댓값과 최솟값 지점 구함.
            iqr = quantile_75 - quantile_25
            iqr_weight = iqr * weight
            lowest_val = quantile_25 - iqr_weight
            highest_val = quantile_75 + iqr_weight

            # 최댓값보다 크거나, 최솟값보다 작은 값을 이상치 데이터로 설정하고 Dataframe index 반환
            outlier_index[column] = column_x[(column_x < lowest_val) | (column_x > highest_val)].index
            print("===========IQR Low~High:", lowest_val, "~", highest_val)
        return outlier_index

    def getOutlierIndexBySeasonalDecomposition(self, outlierDetectorConfig):
        """    
        :param outlierDetectorConfig: have period and limit information ex> {"period":60*24, "limit":10}
        :type outlierDetectorConfig: json

        :return outlier_index: Intersection index by each noise index key
        :type: list
        """

        period = outlierDetectorConfig['period']
        limit = outlierDetectorConfig['limit']

        from statsmodels.tsa.seasonal import seasonal_decompose
        data_outlier = dataOutlier.DataOutlier(self.data)
        data_imputed = data_outlier.imputationForOutlierDetection()

        outlier_index={}
        for feature in data_imputed.columns:
            result = seasonal_decompose(data_imputed[feature],model='additive', period = period)
            n = result.seasonal.mean()+result.trend.mean()
            n = abs(n *limit)
            print("Limit Num:", n)
            NoiseIndex = result.resid[abs(result.resid)> n].index
            outlier_index[feature]= NoiseIndex
            import matplotlib.pyplot as plt
            result.plot()
            plt.show()
            
        return outlier_index

    def getDataWithoutUncertainError(self, outlierIndex):
        """    
        :param outlierIndex: Noise Index
        :type outlierIndex: json

        :return outlierIndex: noise index of each column
        :type: json
        """

        result = dataOutlier.getMoreNaNDataByNaNIndex(self.data, outlierIndex)
        return result

    def getOutlierIndexByMLOutlierDetector(self, outlierDetectorConfig):
        """    
        :param outlierDetectorConfig: Config for outlier detection
        :type outlierDetectorConfig: json

        :return outlierIndex: noise index of each column
        :type: json

        Example
        >>> percentile = 99
        >>> AlgorithmList =[ 'IF', 'KDE', 'LOF', 'MoG', 'SR']
        >>> algorithm = AlgorithmList[2]
        >>> config= {'algorithm': algorithm, 'percentile':percentile}#,'alg_parameter': Parameter[algorithm]
        """
        
        data_outlier = dataOutlier.DataOutlier(self.data)
        data_imputed = data_outlier.imputationForOutlierDetection()
        outlierIndex = data_outlier.getOneDetectionResult(data_imputed, outlierDetectorConfig)
 
        return outlierIndex