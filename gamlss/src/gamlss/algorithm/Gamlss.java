/*
  Copyright 2012 by Dr. Vlasios Voudouris and ABM Analytics Ltd
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/
package gamlss.algorithm;

import gamlss.distributions.*;
import gamlss.utilities.Controls;
import gamlss.utilities.MatrixFunctions;
import gamlss.utilities.oi.CSVFileReader;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Hashtable;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;

/**
 * 01/08/2012.
 * @author Dr. Vlasios Voudouris, Daniil Kiose, 
 * Prof. Mikis Stasinopoulos and Dr Robert Rigby.
 *
 */

public final class Gamlss {

	/** Object of RSAlgorithm class. */	
	private RSAlgorithm rs;
	/** Object of CGAlgorithm class. */
	private CGAlgorithm cg;
	
	/**
	 * 
	 * @param y - response variable
	 */
	private Gamlss(final ArrayRealVector y) {
		
		this(y, null, null);
	}
	
	/**
	 * 
	 * @param y - response variable
	 * @param designMatrices - all design matrices 
	 * (for each distribution parameter) together  
	 */
	private Gamlss(final ArrayRealVector y, 
			  final Hashtable<Integer, BlockRealMatrix> designMatrices) {
		
		this(y, designMatrices, null);
	}

	/** This is to emulate the Gamlss algorithm where 
	 * the user specifies response variable vector 
	 * and design matrix.
	 * 
	 * @param y - vector of response variable values 
	 * @param designMatrices - design matrices for each 
	 * of the distribution parameters
	 * @param smoothMatrices - smoother matrices for each 
	 * of the distribution parameters
	 */
	private Gamlss(final ArrayRealVector y, 
				  Hashtable<Integer, BlockRealMatrix> designMatrices, 
				  final HashMap<Integer, BlockRealMatrix> smoothMatrices) {
		
		GAMLSSFamilyDistribution distr = null;
		switch (DistributionSettings.DISTR) {
	       case DistributionSettings.NO:
	    	   distr = new NO();
	    	   distr.initialiseDistributionParameters(y);
	        break;
	       case DistributionSettings.TF:
	    	   distr = new TF();
	    	   distr.initialiseDistributionParameters(y);
		        break;
	       case DistributionSettings.GA:
	    	   distr = new GA();
	    	   distr.initialiseDistributionParameters(y);
		        break;
	       case DistributionSettings.GT:
	    	   distr = new GT();
	    	   distr.initialiseDistributionParameters(y);
		       break;
	       case DistributionSettings.ST3:
	    	   distr = new ST3();
	    	   distr.initialiseDistributionParameters(y);
		       break;
	       case DistributionSettings.ST4:
	    	   distr = new ST4();
	    	   distr.initialiseDistributionParameters(y);
		       break;
	       case DistributionSettings.JSUo:
	    	   distr = new JSUo();
	    	   distr.initialiseDistributionParameters(y);
		       break;
	       case DistributionSettings.TF2:
	    	   distr = new TF2();
	    	   distr.initialiseDistributionParameters(y);
		       break;
	       case DistributionSettings.SST:
	    	   distr = new SST();
	    	   distr.initialiseDistributionParameters(y);
		       break;
	       case DistributionSettings.BCPE:
	    	   distr = new BCPE();
	    	   distr.initialiseDistributionParameters(y);
		       break;
	       case DistributionSettings.ST1:
	    	   distr = new ST1();
	    	   distr.initialiseDistributionParameters(y);
		       break;
	       case DistributionSettings.PE:
	    	   distr = new PE();
	    	   distr.initialiseDistributionParameters(y);
		       break;
	       default: 
				System.err.println("The specific distribution " 
						+ "has not been implemented yet in Gamlss!");
		}
		
		if (smoothMatrices != null) {
			Controls.SMOOTHING = true;
		}
		
		if (designMatrices == null) {
			designMatrices = new Hashtable<Integer, BlockRealMatrix>();
			for (int i = 1; i < distr.getNumberOfDistribtionParameters() + 1; i++) {
				designMatrices.put(i, 
						MatrixFunctions.buildInterceptMatrix(y.getDimension()));
				Controls.NO_INTERCEPT[i-1] = true;
			}
		}
		
		ArrayRealVector w = new ArrayRealVector(y.getDimension());
		for (int i = 0; i < w.getDimension(); i++) {
					w.setEntry(i, Controls.DEFAULT_WEIGHT);
		}

	
		
		switch (DistributionSettings.FITTING_ALG) {
	      case DistributionSettings.RS:
	    	rs = new RSAlgorithm(distr, y, designMatrices, smoothMatrices, w);
	    	rs.functionRS(distr, y, w);
	        break;
	      case DistributionSettings.RS20CG20:
	    	rs = new RSAlgorithm(distr, y, designMatrices, smoothMatrices, w);
	    	rs.functionRS(distr, y, w);
	    	rs = null;	    	
	    	cg = new CGAlgorithm(y.getDimension());
	    	cg.setnCyc(Controls.GAMLSS_NUM_CYCLES);
	    	cg.CGfunction(distr, y, designMatrices, w);
	        break;	        	        
	      case DistributionSettings.CG:
	    	cg = new CGAlgorithm(y.getDimension());
	    	cg.CGfunction(distr, y, designMatrices, w);
	        break;	        
	      case DistributionSettings.GO:
	           break;           
		  default: 
					System.err.println(" Cannot recognise the " 
												+ "fitting algorithm");
		}
		
		ArrayRealVector tempoM 
			= distr.getDistributionParameter(DistributionSettings.MU);
		ArrayRealVector tempoS 
			= distr.getDistributionParameter(DistributionSettings.SIGMA);
		ArrayRealVector tempoN 
			= distr.getDistributionParameter(DistributionSettings.NU);
		ArrayRealVector tempoT 
			= distr.getDistributionParameter(DistributionSettings.TAU);
		
		System.out.println("Fitted Values:  ");
		System.out.println();

		if (Controls.PRINT) {
			for (int i = 0; i < tempoM.getDimension(); i++) {
			   System.out.println(tempoM.getEntry(i) + " " + tempoS.getEntry(i)
						+ " " + tempoN.getEntry(i) + " " + tempoT.getEntry(i));
			}
		}

		int parNum = distr.getNumberOfDistribtionParameters();
		BlockRealMatrix printMatrix = new BlockRealMatrix(y.getDimension(), parNum);
		for (int i = 0; i < y.getDimension(); i++) {
			for (int j = 0; j < parNum; j++) {
				printMatrix.setEntry(i, j, 
						distr.getDistributionParameter(j+1).getEntry(i));
			}
		}
		
		MatrixFunctions.matrixWriteCSV(
				"C:\\Users\\Daniil\\Desktop\\Gamlss_exp/outJ.csv", printMatrix);
		System.out.println("Done !!!");
	}
	
	/**
	 * Main method.
	 * @param args -  command-line arguments
	 */
	 public static void main(final String[] args) {

		 //String fileName = "Data/dataReduced.csv";
		 String fileName = "Data/data.csv";
		// String fileName = "Data/sp.csv";
		 //String fileName = "Data/dataReduced.csv";
		 CSVFileReader readData = new CSVFileReader(fileName);
		 readData.readFile();
		 ArrayList<String> data = readData.storeValues;
		 
		 ArrayRealVector y = new ArrayRealVector(data.size());
		 BlockRealMatrix muX = new BlockRealMatrix(data.size(), 1);
		 BlockRealMatrix sigmaX = new BlockRealMatrix(data.size(), 1);
		 BlockRealMatrix nuX = new BlockRealMatrix(data.size(), 1);
		 BlockRealMatrix tauX = new BlockRealMatrix(data.size(), 1); 
		 ArrayRealVector w = new ArrayRealVector(data.size());
		
		 BlockRealMatrix muS = new BlockRealMatrix(data.size(), 1);
		 BlockRealMatrix sigmaS = new BlockRealMatrix(data.size(), 1);
		 BlockRealMatrix nuS = new BlockRealMatrix(data.size(), 1);
		 BlockRealMatrix tauS = new BlockRealMatrix(data.size(), 1); 
		
		 for (int i = 0; i < data.size(); i++) {
			String[] line = data.get(i).split(",");
			y.setEntry(i,  Double.parseDouble(line[0]));
			muX.setEntry(i, 0, Double.parseDouble(line[1]));
			muS.setEntry(i, 0, Double.parseDouble(line[1]));
			sigmaX.setEntry(i, 0, Double.parseDouble(line[1]));
			sigmaS.setEntry(i, 0, Double.parseDouble(line[1]));
			nuX.setEntry(i, 0, Double.parseDouble(line[1]));
			nuS.setEntry(i, 0, Double.parseDouble(line[1]));
			tauX.setEntry(i, 0, Double.parseDouble(line[1]));
			tauS.setEntry(i, 0, Double.parseDouble(line[1]));
		 }	
		 
		 Hashtable<Integer, BlockRealMatrix> designMatrices 
		 						= new Hashtable<Integer, BlockRealMatrix>();	
		 designMatrices.put(DistributionSettings.MU, muX);
		 designMatrices.put(DistributionSettings.SIGMA, sigmaX);
		 designMatrices.put(DistributionSettings.NU, nuX);
		 designMatrices.put(DistributionSettings.TAU, tauX);
		 
		 HashMap<Integer, BlockRealMatrix> smoothMatrices 
		 						= new HashMap<Integer, BlockRealMatrix>();
		 smoothMatrices.put(DistributionSettings.MU, muS);
		 smoothMatrices.put(DistributionSettings.SIGMA, sigmaS);
		 smoothMatrices.put(DistributionSettings.NU, nuS);
		 smoothMatrices.put(DistributionSettings.TAU, tauS);
		 
		 //smoothMatrices.put(DistributionSettings.MU, null);
		 //smoothMatrices.put(DistributionSettings.SIGMA, null);
		 //smoothMatrices.put(DistributionSettings.NU, null);
		 //smoothMatrices.put(DistributionSettings.TAU, null);

		 DistributionSettings.DISTR = DistributionSettings.PE;
	
		 //Controls.SMOOTHER = Controls.PB;
		 Controls.SMOOTHER = Controls.RW;
		 
		 
		 Controls.IS_SVD = true;
		 Controls.BIG_DATA = true;
		 Controls.JAVA_OPTIMIZATION = false;

		 new Controls();
		 //Gamlss gamlss = new Gamlss(y, designMatrices, null);
		// Gamlss gamlss = new Gamlss(y, designMatrices, smoothMatrices);
		 Gamlss gamlss = new Gamlss(y, null, smoothMatrices);
	 }
}
