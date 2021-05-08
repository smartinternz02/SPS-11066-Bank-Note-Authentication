package org.ml;

import java.util.Arrays;

import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.components.Figure;
import tech.tablesaw.plotly.components.Layout;
import tech.tablesaw.plotly.traces.BoxTrace;
import tech.tablesaw.plotly.traces.HistogramTrace;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class DataAnalysis {
	
	public static Instances getInstances (String filename)
	{		
		DataSource source;
		Instances dataset = null;
		try {
			source = new DataSource(filename);
			dataset = source.getDataSet();
			dataset.setClassIndex(dataset.numAttributes()-1);
			
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			
		}
		
		return dataset;
	}
	
	@SuppressWarnings("static-access")
	public static void main(String[] args) throws Exception{
		System.out.println("Data Analysis");
	
		Table data = Table.read().csv("K:\\Ritin\\Eclipse\\workspace\\org.ml\\src\\main\\java\\org\\ml\\data_banknote_authentication.csv");
		System.out.println(data.shape());
		System.out.println(data.first(5));
		System.out.println(data.last(5));
		System.out.println(data.structure());
		System.out.println(data.summary());
		
		//Histogram
		Layout layout1 = Layout.builder().title("Distribution of Skewness").build();
		HistogramTrace t1 = HistogramTrace.builder(data.nCol("Skewness")).build();
		Plot.show(new Figure(layout1,t1));
		
		Layout l2 = Layout.builder().title("Class Distribution").build();
		BoxTrace t2 = BoxTrace.builder(data.categoricalColumn("Class"),data.nCol("Skewness")).build();
		Plot.show(new Figure(l2,t2));
		System.out.println("-------------------------------------------------------------------------------------\n\n");
		
		Instances test_data = getInstances("K:\\Ritin\\Eclipse\\workspace\\org.ml\\src\\main\\java\\org\\ml\\testdata.arff");
		Instances train_data = getInstances("K:\\Ritin\\Eclipse\\workspace\\org.ml\\src\\main\\java\\org\\ml\\traindata.arff");
		System.out.println(train_data.size());
		
		//Building a classifier
		Classifier classifier = new weka.classifiers.functions.Logistic();
	
		classifier.buildClassifier(train_data);
		
		
		/**
		 * train the algorithm with the training data and evaluate the
		 * algorithm with testing data
		 */
		Evaluation eval = new Evaluation(train_data);
		eval.evaluateModel(classifier, test_data);
		/** Print the algorithm summary */
		System.out.println("* Logistic Regression Evaluation with Datasets *");
		System.out.println(eval.toSummaryString());
       //System.out.print(" the expression for the input data as per algorithm is ");
      //System.out.println(classifier);
		
		double confusion[][] = eval.confusionMatrix();
		System.out.println("Confusion matrix:");
		for (double[] row : confusion)
			System.out.println(	 Arrays.toString(row));
		System.out.println("-------------------");

		System.out.println("Area under the curve");
		System.out.println( eval.areaUnderROC(0));
		System.out.println("-------------------");
		
		System.out.println(eval.getAllEvaluationMetricNames());
		
		System.out.print("Recall :");
		System.out.println(Math.round(eval.recall(1)*100.0)/100.0);
		
		System.out.print("Precision:");
		System.out.println(Math.round(eval.precision(1)*100.0)/100.0);
		System.out.print("F1 score:");
		System.out.println(Math.round(eval.fMeasure(1)*100.0)/100.0);
		
		System.out.print("Accuracy:");
		double acc = eval.correct()/(eval.correct()+ eval.incorrect());
		System.out.println(Math.round(acc*100.0)/100.0);
		
		
		System.out.println("-------------------");
		Instance predicationDataSet = test_data.get(4);
		double value = classifier.classifyInstance(predicationDataSet);
		/** Prediction Output */
		System.out.println("Predicted label:");
		System.out.print(value);
		
		
	}

}