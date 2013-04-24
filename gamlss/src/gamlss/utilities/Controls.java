/*
  Copyright 2012 by Dr. Vlasios Voudouris and ABM Analytics Ltd
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
  
  Dr. Vlasios Voudouris, Daniil Kiose, Prof. Mikis Stasinopoulos and Dr Robert Rigby
*/
package gamlss.utilities;

import org.apache.commons.math3.linear.ArrayRealVector;

/**
 * 01/08/2012
 * @author Dr. Vlasios Voudouris, Daniil Kiose, Prof.
 *  Mikis Stasinopoulos and Dr Robert Rigby.
 *
 */
public final class Controls {
	/** gamlss convergence criterion-tolerance = cc in R GAMLSS. */
	public static double GAMLSS_CONV_CRIT = 0.001;		
	/** gamlss number of cycles. */
	public static int GAMLSS_NUM_CYCLES = 20;
	/** gamlss trace. */
	public static boolean GAMLSS_TRACE = true;
	/** glim step. */
	public static double GLIM_STEP = 1;
	/** glim tolerance. */
	public static double GLIM_TOL = 0.001;
	/** glim convergence criterion-tolerance. */
	public static double GLIM_CONV_CRIT = 0.001;
	/** glim number of cycles. */
	public static int GLIM_NUM_CYCLES = 50;
	/** glim trace. */
	public static boolean GLIM_TRACE = false;
	/** gamlss global deviance tolerance. */
	public static double GLOB_DEVIANCE_TOL = 5;
	/** gamlss iterations. */
	public static double ITERATIONS = 0;
	/** gamlss auto step. */
	public static boolean AUTO_STEP = true;
	/** gamlss save. */
	public static boolean SAVE = true;
	/** back fitting cycles. */
	public static int BACKFIT_CYCLES = 30;
	/** back fitting tolerance. */
	public static double BACKFIT_TOL = 0.001;
	/** back fitting trace. */
	public static boolean BACKFIT_TRACE = false;
	/** gamlss mu.step. */
	public static double MU_STEP = 1;
	/** gamlss sigma.step. */
	public static double SIGMA_STEP = 1;
	/** gamlss nu.step. */
	public static double NU_STEP = 1;
	/** gamlss tau.step. */
	public static double TAU_STEP = 1;
	/** array of steps for each of distribution parameter */
	public static double[] STEP = new double[4];
	/** mu offset value. */	
	public static double MU_OFFSET = 0;
	/** sigma offset value. */
	public static double SIGMA_OFFSET = 0;
	/** nu offset value. */
	public static double NU_OFFSET = 0;
	/** tau offset value. */
	public static double TAU_OFFSET = 0;
	/** array of offsets for each of distribution parameter. */
	public static double[] OFFSET = new double[4];
	/** ML smoothing method. */
	public final static int ML = 0;
	/** ML1 smoothing method. */
	public final static int ML1 = 1;
	/** EM smoothing method. */
	public final static int EM = 2;
	/** GAIC smoothing method. */
	public final static int GAIC = 3;
	/** GCV smoothing method. */
	public final static int GCV = 4;
	/** to be set for desired smoothing method, default: ML. */
	public static int SMOOTH_METHOD = ML;
	/** MU degree of freedom manually set, default is null.*/
	public static Integer MU_DF = null;
	/** SIGMA degree of freedom manually set, default is null.*/
	public static Integer SIGMA_DF = null;
	/** NU degree of freedom manually set, default is null.*/
	public static Integer NU_DF = null;
	/** TAU degree of freedom manually set, default is null.*/
	public static Integer TAU_DF = null;
	/** User defined array of values of degree 
	 * of freedom for all distr parameters.*/
	public static Integer[] DF_USER_DEFINED = new Integer[4];
	/** is the number of equal space intervals in colValues
	 *  (unless quantiles = TRUE is used in which case the 
	 *  points will be at the quantiles values of colValues) */
	public static int INTER = 20;
	/** is the degree of the polynomial. */
	public static int DEGREE = 3;
	/** whether quantiles (points taken at regular intervals
	 *  from the cumulative distribution function (CDF) 
	 *  of a random variable) used or not */
	public static boolean QUANTILES = false;
	/** order refers to differences in the penalty for the 
	 * coefficients, order = 0 : white noise random effects ,
	 *  order = 1 : random walk, order = 2 : random walk of order 2 , 
	 *  order = 3 : random walk of order 3.*/
	public static int ORDER = 2;
	/** User defined value of lambda. */
	public static Double LAMBDAS_USER_DEFINED = null;
	/** xevalUserDefined = null if no prediction used. */
	public static ArrayRealVector XEVAL_USER_DEFINED = null;
	/** whether to copy the original data. */
	public static boolean COPY_ORIGINAL = true;
	/** initial value of lambda. */
	public static double INITIAL_LAMBDA = 10.0;
	/** this part of algorithm is very expensive but results never used. */
	public static boolean IF_NEED_THIS = false;
	/** boolean to specify whether Mu will be 
	 * fitted with or without an intercept term. */
	public static boolean NO_INTERCEPT_MU = false;
	/** boolean to specify whether Sigma will be 
	 * fitted with or without an intercept term. */
	public static boolean NO_INTERCEPT_SIGMA = false;
	/** boolean to specify whether Nu will be 
	 * fitted with or without an intercept term. */
	public static boolean NO_INTERCEPT_NU = false;
	/** boolean to specify whether Tau will be 
	 * fitted with or without an intercept term. */
	public static boolean NO_INTERCEPT_TAU = false;
	/** Array of booleans to specify whether the 
	 * model will be fitted with or without an intercept term. */
	public static boolean[] NO_INTERCEPT = new boolean[4];
	/** default weights. */
	public static double DEFAULT_WEIGHT = 1.0;
	/** whether to use SVD decomposition or QR in linear fitting alg.  */
	public static boolean IS_SVD = false;
	/**   */
	public static boolean IS_SE = false;
	/** splitTolerance  - tolerance on the off-diagonal
	 *  elements relative to the geometric mean to split 
	 *  the tridiagonal matrix.*/
	public static double SPLIT_TOLERANCE = 2.2250738585072014E-308;
	/**   */
	public static Integer K = 2;
	/** allowed number of evaluations of the objective function. */
	public static int BOBYQA_MAX_EVAL = 200;
	/** allowed number of algorithm iterations. */
	public static int BOBYQA_MAX_ITER = 150;
	/**  number of interpolation conditions. For a problem of dimension
	 *  n, its value must be in the interval [n+2, (n+1)(n+2)/2]. 
	 *  Choices that exceed 2n+1 are not recommended */
	public static int BOBYQA_INTERPOL_POINTS = 5;
	/** allowed number of iterations in ML1 smoothing method */
	public static int ML1_ITER = 50;
	/** allowed number of iterations in ML smoothing method */
	public static int ML_ITER = 50;
	/** PB smoother */
	public final static int PB = 1;
	/** Random walk smoother */
	public final static int RW = 2;
	/** Smoother algorithm: default PB */
	public static int SMOOTHER = PB;
	/** order refers to differences in the penalty for the 
	 * coefficients, order = 0 : white noise random effects ,
	 *  order = 1 : random walk, order = 2 : random walk of order 2 , 
	 *  order = 3 : random walk of order 3. Default value of order in RW is 1.*/
	public static int RW_ORDER = 1;
	/** Initial value of SIG2E. */
	public static double RW_SIG2E = 1.0;
	/** Initial value of SIG2B. */
	public static double RW_SIG2B = 1.0;
	/** Initial values of DELTA used in RW. */
	public static double[] RW_DELTA = {0.01, 0.01};
	/** Initial values of SHIFT used in RW. */
	public static double[] RW_SHIFT = {0.0, 0.0};
	/** Whether smoothing is used - becomes true when 
	 * smoothing matrices are supplied to Gamlss */
	public static boolean SMOOTHING = false;
	/** Boolean - whether SIG2E is fixed or not  */
	public static boolean SIG2E_FIX = false;
	/** Boolean - whether SIG2B is fixed or not  */
	public static boolean SIG2B_FIX = false;
	/** Boolean - whether to use a penalised regression. Default is false.*/
	public static boolean PENALTY = false;
	/** This controls whether to use only Apache library for matrix operations (reasonable for
	 *  the small data sets only) or connect R to handle huge matrices. 
	 *  Default is false - use Apache. */
	public static boolean BIG_DATA = false;
	
	public static boolean JAVA_OPTIMIZATION = true;
	
	public static boolean PRINT = false;
	
	public static boolean LOG_LIKELIHOOD = true;
	
	public Controls() {
		
		STEP[0] = MU_STEP;
		STEP[1] = SIGMA_STEP;
		STEP[2] = NU_STEP;
		STEP[3] = TAU_STEP;
		
		OFFSET[0] = MU_OFFSET;
		OFFSET[1] = SIGMA_OFFSET;
		OFFSET[2] = NU_OFFSET;
		OFFSET[3] = TAU_OFFSET;
		
		DF_USER_DEFINED[0] = MU_DF;
		DF_USER_DEFINED[1] = SIGMA_DF;
		DF_USER_DEFINED[2] = NU_DF;
		DF_USER_DEFINED[3] = TAU_DF;
		
		NO_INTERCEPT[0] = NO_INTERCEPT_MU;
		NO_INTERCEPT[1] = NO_INTERCEPT_SIGMA;
		NO_INTERCEPT[2] = NO_INTERCEPT_NU;
		NO_INTERCEPT[3] = NO_INTERCEPT_TAU;
	}
}
