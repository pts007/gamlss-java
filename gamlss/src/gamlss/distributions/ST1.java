/*
  Copyright 2012 by Dr. Vlasios Voudouris and ABM Analytics Ltd
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/
package gamlss.distributions;

import gamlss.distributions.GT.IntegratingFunction;
import gamlss.distributions.GT.UniRootObjFunction;
import gamlss.utilities.Controls;
import gamlss.utilities.MakeLinkFunction;
import gamlss.utilities.TDistr;

import java.util.Hashtable;

import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.integration.BaseAbstractUnivariateIntegrator;
import org.apache.commons.math3.analysis.integration.LegendreGaussIntegrator;
import org.apache.commons.math3.analysis.integration.UnivariateIntegrator;
import org.apache.commons.math3.analysis.solvers.BrentSolver;
import org.apache.commons.math3.analysis.solvers.UnivariateSolver;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.TDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.exception.TooManyEvaluationsException;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.util.FastMath;

/**
 * @author Dr. Vlasios Voudouris, Daniil Kiose, 
 * Prof. Mikis Stasinopoulos and Dr Robert Rigby.
 */
public class ST1 implements GAMLSSFamilyDistribution {
	
	/** Number of distribution parameters. */
	private final int numDistPar = 4;
	/** Hashtable to hold vectors of distribution 
	 * parameters values (mu, sigma, ...). */
	private Hashtable<Integer, ArrayRealVector> distributionParameters 
							= new Hashtable<Integer, ArrayRealVector>();
	/** Hashtable to hold types of link functions 
	 * for the distribution parameters. */
	private Hashtable<Integer, Integer> distributionParameterLink 
							= new Hashtable<Integer, Integer>();
	/** vector of values of mu distribution parameter. */
	private ArrayRealVector muV;
	/** vector of values of sigma distribution parameter. */
	private ArrayRealVector sigmaV;
	/** vector of values of nu distribution parameter. */
	private ArrayRealVector nuV;
	/** vector of values of tau distribution parameter. */
	private ArrayRealVector tauV;
	/** Array of first derrivative values dl/dmu. */
	private double[] dldm;
	/** Array of first derrivative values dl/dsigma. */
	private double[] dlds;
	/** Array of first derrivative values dl/dnu. */
	private double[] dldn;
	/** Array of first derrivative values dl/dtau. */
	private double[] dldt;
	/** Temporary vector for interim operations. */
	private ArrayRealVector tempV;	
	/** Temporary array for interim operations. */
	private double[] z;
	/** Temporary array for interim operations. */
	private double[] w;
	/** Temporary array for interim operations. */
	private double[] lam;
	/** Temporary int for interim operations. */
	private int size;
	/** Object of t distribution class. */
	private TDistr tdDist;
	/** Object of Normal distribution class. */
	private NormalDistribution noDist;
	/** Object of LegendreGaussIntegrator class. */
	private LegendreGaussIntegrator integrator;
	/** Object of IntegratingFunction class. */
	private IntegratingFunction function;
	/** Temporary array for interim operations. */
	private double[] interval;
	/** object of uni-root objective function. */
	private UniRootObjFunction uniRootObj;
	/** Object of UnivariateSolver class. */
	private UnivariateSolver uniRootSolver;
	
	/** This is the Skew t (Azzalini type 1)  distribution
	 *  with default link (muLink="identity",sigmaLink="log",
	 *   nuLink="log", tauLink="log"). */
	public ST1() {
	
	this(DistributionSettings.IDENTITY, 
		 DistributionSettings.LOG, 
		 DistributionSettings.IDENTITY, 
		 DistributionSettings.LOG);
	}
	
	/** This is the Skew t (Azzalini type 1) distribution with
	 *  supplied link function for each of the distribution parameters.
	 * @param muLink - link function for mu distribution parameter
	 * @param sigmaLink - link function for sigma distribution parameter
	 * @param nuLink - link function for nu distribution parameter
	 * @param tauLink - link function for tau distribution parameter
	 */
	public ST1(final int muLink, 
			   final int sigmaLink, 
			   final int nuLink, 
			   final int tauLink) {
	
		distributionParameterLink.put(DistributionSettings.MU,
				 MakeLinkFunction.checkLink(DistributionSettings.ST1, muLink));
		distributionParameterLink.put(DistributionSettings.SIGMA,
			  MakeLinkFunction.checkLink(DistributionSettings.ST1, sigmaLink));
		distributionParameterLink.put(DistributionSettings.NU,
				 MakeLinkFunction.checkLink(DistributionSettings.ST1, nuLink));
		distributionParameterLink.put(DistributionSettings.TAU,
				MakeLinkFunction.checkLink(DistributionSettings.ST1, tauLink));
		
		tdDist = new TDistr();
		noDist = new NormalDistribution();
		integrator  = new LegendreGaussIntegrator(2, 
		        LegendreGaussIntegrator.DEFAULT_RELATIVE_ACCURACY,
					LegendreGaussIntegrator.DEFAULT_ABSOLUTE_ACCURACY);
		function = new IntegratingFunction();
		interval = new double[2];
		uniRootSolver = new BrentSolver(1.0e-12, 1.0e-8);
		uniRootObj = new UniRootObjFunction();
	}

	/** Initialises the distribution parameters.
	 * @param y - response variable */
	public final void initialiseDistributionParameters(
												final ArrayRealVector y) {
		
		distributionParameters.put(DistributionSettings.MU, 
													setMuInitial(y));
		distributionParameters.put(DistributionSettings.SIGMA, 
													setSigmaInitial(y));
		distributionParameters.put(DistributionSettings.NU, 
													setNuInitial(y));
		distributionParameters.put(DistributionSettings.TAU,
													setTauInitial(y));
	}
	
	/**  Calculate and set initial value of mu, by assumption 
	 * these values lie between observed data and the trend line.
	 * @param y - vector of values of response variable
	 * @return vector of initial values of mu
	 */
	private ArrayRealVector setMuInitial(final ArrayRealVector y) {
		//mu.initial =  expression(mu <- (y+mean(y))/2)
		size = y.getDimension();
		double[] out = new double[size];
		Mean mean = new Mean();	
		double yMean = mean.evaluate(y.getDataRef());	
		for (int i = 0; i < size; i++) {
			out[i] = (y.getEntry(i) + yMean) / 2;
		}
		return new ArrayRealVector(out, false);
	}
	

	/** Calculate and set initial value of sigma.
	 * @param y - vector of values of response variable
	 * @return vector of initial values of sigma
	 */
	private ArrayRealVector setSigmaInitial(final ArrayRealVector y) {
		//sigma.initial = expression(sigma <- rep(sd(y)/4, length(y))),
		tempV = new ArrayRealVector(y.getDimension());
		final double out = new StandardDeviation().evaluate(y.getDataRef());
		tempV.set(out / 4);
		return tempV;
	}

	
	/** Calculate and set initial value of nu.
	 * @param y - vector of values of response variable
	 * @return vector of initial values of nu
	 */
	private ArrayRealVector setNuInitial(final ArrayRealVector y) {	
		//nu.initial = expression(nu <- rep(0.1, length(y))),
		tempV = new ArrayRealVector(y.getDimension());
		tempV.set(0.1);
		return tempV;
	}
	
	


	/** Calculates initial value of tau.
	 * @param y - vector of values of response variable
	 * @return vector of initial values of tau
	 */
	private ArrayRealVector setTauInitial(final ArrayRealVector y) {
		//tau.initial = expression(tau <-rep(5, length(y)))
		tempV = new ArrayRealVector(y.getDimension());
		tempV.set(5.0);
		return tempV;
	}

	
/** Calculates a first derivative of the likelihood 
	 * function in respect to supplied distribution parameter.
	 * @param whichDistParameter - distribution parameter
	 * @param y - vector of values of likelihood function
	 * @return vector of first derivative of the likelihood 
	 */
	public final ArrayRealVector firstDerivative(final int whichDistParameter, 
										   	     final ArrayRealVector  y) {
		setInterimArrays(y, whichDistParameter);
		tempV = null;		
		switch (whichDistParameter) {
	    case DistributionSettings.MU:
	    	tempV = dldm(y);
	       break;
	    case DistributionSettings.SIGMA:
	    	tempV = dlds(y);
	      break;
	    case DistributionSettings.NU:
	    	tempV = dldn(y);
	      break;
	    case DistributionSettings.TAU:
	    	tempV = dldt(y);
	      break;
		default: 
			System.err.println("Requested first order "
					+ "derivative does not exist");
			break;
	    }			
		return tempV;
	}
	
	/**
	 * Set z, w, lam arrays.
	 * @param y - response variable
	 */
	private void setInterimArrays(final ArrayRealVector y,
								  final int whichDistParameter) {
		muV     = distributionParameters.get(DistributionSettings.MU);
		sigmaV  = distributionParameters.get(DistributionSettings.SIGMA);
	 	nuV     = distributionParameters.get(DistributionSettings.NU);
	 	tauV    = distributionParameters.get(DistributionSettings.TAU);
		
	 	size = y.getDimension();
		z = new double[size];
		lam = new double[size];
		w = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		
	 		//z <- (y-mu)/sigma
	 		z[i] = (y.getEntry(i) - muV.getEntry(i)) / sigmaV.getEntry(i);
	 		
	 	    //w <- nu*z
	 		w[i] = nuV.getEntry(i) * z[i];
	 		
	 		if (whichDistParameter != DistributionSettings.NU) {
	 			//lam <- ifelse(tau < 1000000, (tau+1)/(tau+(z^2)),1)
	 			if (tauV.getEntry(i) < 1000000) {
	 			
	 				lam[i] = (tauV.getEntry(i) + 1) 
	 							/ (tauV.getEntry(i) + (z[i] * z[i]));
	 			} else {
	 			
	 				lam[i] = 1.0;
	 			}
	 		}
	 	}
	}

	/**  First derivative dldm = dl/dmu, where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of first derivative dldm = dl/dmu
	 */	
	public final ArrayRealVector dldm(final ArrayRealVector y) { 

	 	dldm = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		
	 		//dldm <- -(dt(w,df=tau)/pt(w,df=tau))*nu/sigma + lam*z/sigma
	 		tdDist.setDegreesOfFreedom(tauV.getEntry(i));
	 		dldm[i] = -(tdDist.density(w[i]) 
	 				/ tdDist.cumulativeProbability(w[i])) 
	 				* nuV.getEntry(i) / sigmaV.getEntry(i) 
	 					+ lam[i] * z[i] / sigmaV.getEntry(i);
	 	}
		z = null;
		w = null;
		lam = null;
	  	return new ArrayRealVector(dldm, false);
	}
	
	/** First derivative dlds = dl/dsigma, where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of First derivative dlds = dl/dsigma
	 */	
	public final ArrayRealVector dlds(final ArrayRealVector y) {	
		
		dlds = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		
	 		//dldd <- -(dt(w,df=tau)/pt(w,df=tau))*nu*z/sigma 
	 		//+ ((lam*(z^2))-1)/sigma
	 		tdDist.setDegreesOfFreedom(tauV.getEntry(i));
	 		dlds[i] = -(tdDist.density(w[i]) 
	 				/ tdDist.cumulativeProbability(w[i])) 
	 				* nuV.getEntry(i) * z[i] / sigmaV.getEntry(i) 
	 				+ ((lam[i] * (z[i] * z[i])) - 1) / sigmaV.getEntry(i);
	 	}
		z = null;
		w = null;
		lam = null;
	  	return new ArrayRealVector(dlds, false);
	}

	/** First derivative dldn = dl/dnu, where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of First derivative dldn = dl/dnu
	 */	
	public final ArrayRealVector dldn(final ArrayRealVector y) {	

		dldn = new double[size];
		double temp = 0;
	 	for (int i = 0; i < size; i++) {
	 		
	 		//dwdv <- w/nu
	 		//dldv <- (dt(w,df=tau)/pt(w,df=tau))*dwdv
	 		tdDist.setDegreesOfFreedom(tauV.getEntry(i));
	 		dldn[i] = (tdDist.density(w[i]) 
	 				/ tdDist.cumulativeProbability(w[i])) 
	 								* w[i] / nuV.getEntry(i);
	 	 }
		z = null;
		w = null;
		lam = null;
	  	return new ArrayRealVector(dldn, false);
	}
	
	/** First derivative dldtau = dl/dtau, where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of First derivative dldtau = dl/dtau
	 */
	public final ArrayRealVector dldt(final ArrayRealVector y) {	
		
		dldt = new double[size];
	 	for (int i = 0; i < size; i++) {
 		
 		//j <- (pt(w,df=tau+0.001,log.p=TRUE)
	 	//-pt(w,df=tau-0.001,log.p=TRUE))/0.002
	 	tdDist.setDegreesOfFreedom(tauV.getEntry(i) + 0.001);
	 	final double part1 = FastMath.log(tdDist.cumulativeProbability(w[i]));
	 	
	 	tdDist.setDegreesOfFreedom(tauV.getEntry(i) - 0.001);	
	 	final double part2 = FastMath.log(tdDist.cumulativeProbability(w[i]));
	
	 	final double j = (part1 - part2) / 0.002;

 		//dldt <- j + (digamma((tau+1)/2)-digamma(tau/2)
	 	//-(1/tau)-log(1+(z^2)/tau)+lam*(z^2)/tau)/2
 		dldt[i] = j + (Gamma.digamma((tauV.getEntry(i) + 1) 
 				/ 2) - Gamma.digamma(tauV.getEntry(i) / 2) 
 				- (1/tauV.getEntry(i)) - FastMath.log(1 + (z[i] 
 						* z[i]) / tauV.getEntry(i)) + lam[i] * (z[i] 
 									* z[i]) / tauV.getEntry(i)) / 2.0;
 	 }
		z = null;
		w = null;
		lam = null;
	  	return new ArrayRealVector(dldt, false);
}
	
/** Calculates a second derivative of the likelihood 
	 * function in respect to supplied distribution parameter.
	 * @param whichDistParameter - distribution parameter
	 * @param y - vector of values of likelihood function 
	 * @return vector of second derivative of the likelihood
	 */
	public final ArrayRealVector secondDerivative(final int whichDistParameter,
												  final ArrayRealVector y) {
				tempV = null;		
				switch (whichDistParameter) {
			      case DistributionSettings.MU:
			    	   tempV = d2ldm2(y);
			          break;
			      case DistributionSettings.SIGMA:
			    	 tempV = d2lds2(y);
			    	 break;
			      case DistributionSettings.NU:
			    	 tempV = d2ldn2(y);
			    	 break;
			      case DistributionSettings.TAU:
			    	 tempV = d2ldt2(y);
			    	 break;
			      default: 
					System.err.println("Requested second order "
							+ "derivative does not exist");
					break;
			    }
				return tempV;
	}

	/** Second derivative d2ldm2= (d^2l)/(dmu^2),
	 *  where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of second derivative d2ldm2= (d^2l)/(dmu^2)
	 */
	private ArrayRealVector d2ldm2(final ArrayRealVector y) {	

		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		//d2ldm2 <- -dldm*dldm
	 		out[i] = -dldm[i] * dldm[i];
		 	if (out[i] > -1e-15) {
		 		out[i] = -1e-15;
		 	}
	 	}
	 	muV     = null;
	 	sigmaV  = null;
	 	nuV     = null;
	 	tauV    = null;
	  	return new ArrayRealVector(out, false);
	}
	
	/** Second derivative d2lds2= (d^2l)/(dsigma^2), 
	 * where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of second derivative d2lds2= (d^2l)/(dsigma^2)
	 */
	private ArrayRealVector d2lds2(final ArrayRealVector y) {
	
		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		//d2ldd2 <- -dldd*dldd
	 		out[i] = -dlds[i] * dlds[i];
		 	if (out[i] > -1e-15) {
		 		out[i] = -1e-15;
		 	}
	 	 }
	 	 muV     = null;
	 	 sigmaV  = null;
	 	 nuV     = null;
	 	 tauV    = null;
	  	return new ArrayRealVector(out, false);

	}
	
	/** Second derivative d2ldn2= (d^2l)/(dnu^2), 
	 * where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of second derivative d2ldn2= (d^2l)/(dnu^2)
	 */
	private ArrayRealVector d2ldn2(final ArrayRealVector y) {	

		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		//d2ldv2 <- -dldv*dldv
	 		out[i] = -dldn[i] * dldn[i];
	 		
	 		//d2ldv2 <- ifelse(d2ldv2 < -1e-15, d2ldv2,-1e-15)
		 	if (out[i] > -1e-15) {
		 		out[i] = -1e-15;
		 	}
	 	 }
	 	 muV     = null;
	 	 sigmaV  = null;
	 	 nuV     = null;
	 	 tauV    = null;
	  	return new ArrayRealVector(out, false);
	}
	
	/** Second derivative d2ldt2= (d^2l)/(dtau^2),
	 * where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of second derivative d2ldt2= (d^2l)/(dtau^2)
	 */ 
	private ArrayRealVector d2ldt2(final ArrayRealVector y) {	
		
		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		//d2ldt2 <- -dldt*dldt
	 		out[i] = -dldt[i] * dldt[i];
	 		
	 		//d2ldt2 <- ifelse(d2ldt2 < -1e-15, d2ldt2,-1e-15)
		 	if (out[i] > -1e-15) {
		 		out[i] = -1e-15;
		 	}
	 	 }
	 	 muV     = null;
	 	 sigmaV  = null;
	 	 nuV     = null;
	 	 tauV    = null;
	  	return new ArrayRealVector(out, false);
	}
	
	/** Calculates a second cross derivative of the likelihood 
	 * function in respect to supplied distribution parameters.
	 * @param whichDistParameter1 - first distribution parameter
	 * @param whichDistParameter2 - second distribution parameter
	 * @param y - vector of values of likelihood function 
	 * @return  vector of second cross derivative of the likelihood
	 */
	public final ArrayRealVector secondCrossDerivative(
												 final int whichDistParameter1,
												 final int whichDistParameter2, 
												 final ArrayRealVector y) {
		tempV = null;		
		if (whichDistParameter1 == DistributionSettings.MU) {
			switch (whichDistParameter2) {
			case DistributionSettings.SIGMA:
				tempV = d2ldmds(y);                      
				break;
			case DistributionSettings.NU:
				tempV = d2ldmdn(y);
				break;
			case DistributionSettings.TAU:
				tempV = d2ldmdt(y);
				break;
	          default: 
		  		System.err.println("Second derivative does not exist");
		  		return null;
			}
		}
		if (whichDistParameter1 == DistributionSettings.SIGMA) {
			switch (whichDistParameter2) {
			case DistributionSettings.NU:
				tempV = d2ldsdn(y);
				break;
			case DistributionSettings.TAU:
				tempV = d2ldsdt(y);
				break;
	          default: 
		  		System.err.println("Second derivative does not exist");
		  		return null;
			}
		}
		if (whichDistParameter1 == DistributionSettings.NU) {
			switch (whichDistParameter2) {
			case DistributionSettings.TAU:
				tempV = d2ldndt(y);
				break;
	          default: 
	  			System.err.println("Second derivative does not exist");
	  			return null;
			}
			
		}
		return tempV;
		}
	
/** Second cross derivative of likelihood function in 
	 * respect to mu and sigma (d2ldmdd = d2l/dmu*dsigma).
	 * @param y - vector of values of response variable
	 * @return  a vector of Second cross derivative
	 */
	private ArrayRealVector d2ldmds(final ArrayRealVector y) {	
		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		// d2ldmdd <- -(dldm*dldd)
	 		out[i] = -dldm[i] * dlds[i];
	 	}
	  	return new ArrayRealVector(out, false);
	}

	/** Second cross derivative of likelihood function
 	 * in respect to mu and nu (d2ldmdd = d2l/dmu*dnu).
	 * @param y - vector of values of response variable
	 * @return  a vector of Second cross derivative
	 */
	private ArrayRealVector d2ldmdn(final ArrayRealVector y) {
		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		//d2ldmdv <- -(dldm*dldv)
	 		out[i] = -dldm[i] * dldn[i];
	 	}
	  	return new ArrayRealVector(out, false);
	}	
	
	/** Second cross derivative of likelihood function 
	 * in respect to mu and tau (d2ldmdd = d2l/dmu*dtau).
	 * @param y - vector of values of response variable
	 * @return  a vector of Second cross derivative
	 */
	private ArrayRealVector d2ldmdt(final ArrayRealVector y) {	
		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		//d2ldmdt <- -(dldm*dldt)
	 		out[i] = -dldm[i] * dldt[i];
	 	}
	  	return new ArrayRealVector(out, false);
	}  	
	
	/** Second cross derivative of likelihood function 
	 * in respect to sigma and nu (d2ldmdd = d2l/dsigma*dnu).
	 * @param y - vector of values of response variable
	 * @return  a vector of Second cross derivative
	 */
	private ArrayRealVector d2ldsdn(final ArrayRealVector y) {	
		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		//d2ldddv <- -(dldd*dldv)
	 		out[i] = -dlds[i] * dldn[i];
	 	}
	  	return new ArrayRealVector(out, false);
	}
	
	/** Second cross derivative of likelihood function 
	 * in respect to sigma and tau (d2ldmdd = d2l/dsigma*dtau).
	 * @param y - vector of values of response variable
	 * @return  a vector of Second cross derivative
	 */
	private ArrayRealVector d2ldsdt(final ArrayRealVector y) {	
		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		//d2ldddt <- -(dldd*dldt) 
	 		out[i] = -dlds[i] * dldt[i];
	 	}
	  	return new ArrayRealVector(out, false);
	}	
	

	/** Second cross derivative of likelihood function 
	 * in respect to nu and tau (d2ldmdd = d2l/dnu*dtau).
	 * @param y - vector of values of response variable
	 * @return  a vector of Second cross derivative
	 */
	private ArrayRealVector d2ldndt(final ArrayRealVector y) {	
		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		//d2ldvdt <- -(dldv*dldt)
	 		out[i] = -dldn[i] * dldt[i];
	 	}
	  	return new ArrayRealVector(out, false);
	}
  	
	/** Computes the global Deviance Increament.
	 * @param y - vector of response variable values
	 * @return vector of global Deviance Increament values 
	 */
	public final ArrayRealVector globalDevianceIncreament(
													final ArrayRealVector y) {
        //G.dev.incr  = function(y,mu,sigma,nu,tau,...)  
		//-2*dST3(y,mu,sigma,nu,tau,log=TRUE),
		size = y.getDimension();
		double[] out = new double[size];

		double[] muArr = distributionParameters.get(
									DistributionSettings.MU).getDataRef();
		double[] sigmaArr = distributionParameters.get(
									DistributionSettings.SIGMA).getDataRef();
		double[] nuArr = distributionParameters.get(
									DistributionSettings.NU).getDataRef();
		double[] tauArr = distributionParameters.get(
									DistributionSettings.TAU).getDataRef();
		
		for (int i = 0; i < size; i++) {
			
			out[i] = (-2) * dST1(y.getEntry(i), muArr[i], sigmaArr[i], 
								nuArr[i], tauArr[i], Controls.LOG_LIKELIHOOD);
		}
		return new ArrayRealVector(out, false);
	}
	
	/** Computes the probability density function (PDF) of this 
	 * distribution evaluated at the specified point x.
	 * @param x - value of response variable
	 * @param mu - value of mu distribution parameter
	 * @param sigma - value of sigma distribution parameter
	 * @param nu - value of nu distribution parameter
	 * @param tau - value of tau distribution parameter
	 * @param isLog  - logical, whether to take log of the function or not
	 * @return value of probability density function
	 */
	public final double dST1(final double x, 
				 		     final double mu, 
				 		     final double sigma, 
				 		     final double nu, 
				 		     final double tau, 
				 		     final boolean isLog) {

		// {  if (any(sigma <= 0))stop(paste("sigma must be positive",))
		if (sigma <= 0) {
			System.err.println("sigma must be positive");
			return -1.0;
		}
	
		//if (any(tau <= 0))  stop(paste("tau must be positive", "\n", ""))
		if (tau <= 0) {
			System.err.println("tau must be positive");
			return -1.0;
		}
	
		double out = 0;
		
		//z <- (x-mu)/sigma
		final double z = (x - mu) / sigma;
		
	    //w <- nu*z
		final double w = nu * z;

		//loglik <- ifelse(tau<1000000, loglik1, loglik2)
		if(tau < 1000000) {
			
			//loglik1 <- pt(w,df=tau,log.p=TRUE) 
			//+ dt(z,df=tau,log =TRUE) + log(2) - log(sigma)
			tdDist.setDegreesOfFreedom(tau);
			out = FastMath.log(tdDist.cumulativeProbability(w)) 
					+ FastMath.log(tdDist.density(z)) + FastMath.log(2.0)
														- FastMath.log(sigma);
		} else {
			
			//loglik2 <- pNO(w,mu=0,sigma=1,log.p=TRUE) 
			//+ dNO(z,mu=0,sigma=1,log =TRUE) + log(2) - log(sigma)
			out = FastMath.log(noDist.cumulativeProbability(w)) 
					+ FastMath.log(noDist.density(z)) + FastMath. log(2.0) 
														- FastMath.log(sigma);
		}
		
		//if(log==FALSE) ft  <- exp(loglik) else ft <- loglik 
		if(!isLog) {
			out = FastMath.exp(out);
		}
	  	return out;
	}


	/**
	 * dST1(x) launches dST1(x, mu, sigma, nu, isLog) 
	 * with deafult mu = 0, sigma = 1, nu = 0, tau = 2, isLof=false.
	 * @param x - value of response variable 
	 * @return value of probability density function 
	 */
	//dST1 <- function(x, mu = 0, sigma = 1, nu = 0, tau = 2, log = FALSE)
	public final double dST1(final double x) {
		return dST1(x, 0.0, 1.0, 0.0, 2.0, false);
	}

	/** Computes the cumulative distribution 
	 * function P(X <= q) for a random variable X .
	 * whose values are distributed according to this distribution
	 * @param q - value of quantile
	 * @param mu - value of mu distribution parameter
	 * @param sigma - value of sigma distribution parameter
	 * @param nu - value of nu distribution parameter 
	 * @param tau - value of tau distribution parameter 
	 * @param lowerTail - logical, if TRUE (default), probabilities
	 *  are P[X <= x] otherwise, P[X > x].
	 * @param isLog - logical, if TRUE, probabilities p are given as log(p)
	 * @return value of cumulative probability function values P(X <= q)
	 */
	public final double pST1(final double q, 
 	   		                 final double mu, 
 	   		                 final double sigma, 
 	   		                 final double nu,
 	   		                 final double tau, 
 	   		                 final boolean lowerTail, 
 	   		                 final boolean isLog) {
		
		// {  if (any(sigma <= 0))stop(paste("sigma must be positive",))
		if (sigma <= 0) {
			System.err.println("sigma must be positive");
			return -1.0;
		}
	
		//if (any(tau < 0))  stop(paste("tau must be positive", "\n", ""))
		if (tau < 0) {
			System.err.println("tau must be positive");
			return -1.0;
		}
	
		//cdf[i] <- integrate(function(x) 
		//dST1(x, mu = 0, sigma = 1, nu = nu[i], tau = tau[i]),
		//-Inf, (q[i]-mu[i])/sigma[i] )$value
		function.setNu(nu);
		function.setTau(tau);
		double out = integrator.integrate(Integer.MAX_VALUE, function, 
							Double.NEGATIVE_INFINITY, (q - mu) / sigma);
		
		//if(lower.tail==TRUE) cdf  <- cdf else  cdf <- 1-cdf
	    if (!lowerTail) {
	    	if (isLog) {
	    		out = FastMath.log(1 - out);
	    	} else {
	    		out = 1 - out;
	    	}
	    } else if (isLog) {
	    	//if(log.p==FALSE) cdf  <- cdf else  cdf <- log(cdf)
	    	out = FastMath.log(out);
	    }
		return out;
	}

 	/**
  	 * Inner class is a shell for objective function to
  	 * find the root of the function.
  	 *
  	 */
	class IntegratingFunction implements UnivariateFunction {
		
	    /** nu distribution parameter. */  
		private double nu;
		/** tau distribution parameter. */  
		private double tau;		
				
		/**
		 * This function is used to integrate the objectivefunction function.
		 * @param x - income value to determine zero of the function
		 * @return value of the function
		 */
		public double value(final double x) {
			return dST1(x, 0.0, 1.0, nu, tau, false);
		}	
		
		/**
		 * Set nu distribution parameter.
		 * @param nu - distribution parameter
		 */
	    public void setNu(final double nu) {
			this.nu = nu;
		}
	    
		/**
		 * Set tau distribution parameter.
		 * @param tau - distribution parameter
		 */
	    public void setTau(final double tau) {
			this.tau = tau;
		}
	}

	/**
	 * pST1(q) launches pST1(q, mu, sigma, nu,  tau, 
	 * lowerTail, isLog) with deafult mu = 0, sigma = 1, 
	 * nu = 0, tau = 2.
	 * lowerTail = true, isLog = false.
	 * @param q - quantile
	 * @return cumulative probability function value P(X <= q)
	 */
	//  pST1 <- function(q, mu = 0, sigma = 1, nu = 0, 
	//tau = 2, lower.tail = TRUE, log.p = FALSE)
	public final double pST1(final double q) {
		return pST1(q, 0.0, 1.0, 0.0, 2.0, true, false);
	}	
	
	/** Computes the quantile (inverse cumulative probability)
	 *  function  of this distribution.
	* @param p - value of cumulative probability
	* @param mu -  value of mu distribution parameters
	* @param sigma -  value of sigma distribution parameters
	* @param nu -  value of nu distribution parameters 
	* @param tau -  value of tau distribution parameters
	* @param lowerTail - logical; if TRUE (default), probabilities 
	* are P[X <= x] otherwise, P[X > x]
	* @param isLog - logical; if TRUE, probabilities p are given as log(p).
	* @return value of quantile function
	*/
	public final double qST1(final double p, 
	         				 final double  mu, 
	         				 final double  sigma, 
	         				 final double nu, 
	         				 final double tau, 
	         				 final boolean lowerTail, 
	         				 final boolean isLog) {
		
		// {  if (any(sigma <= 0))stop(paste("sigma must be positive",))
		if (sigma <= 0) {
			System.err.println("sigma must be positive");
			return -1.0;
		}
	
		//if (log.p==TRUE) p <- exp(p) else p <- p
		double out = 0;
		double temp = p;
		// if (log.p==TRUE) p <- exp(p) else p <- p
		if (isLog) {
			temp = FastMath.exp(temp);
		}
		if (temp <= 0 || temp >= 1) {
			System.err.println("p must be between 0 and 1");
		}
		if (!lowerTail) {
			temp =  1 - temp;
		}
		
		//if (h(mu[i])<p[i])
		if (h(mu, mu, sigma, nu, tau, true, false) < temp) {
			
			//interval <- c(mu[i], mu[i]+sigma[i])
			interval[0] = mu;
			interval[1] = mu + sigma;
			
			//j <-2
			int j = 2;
			
			//while (h(interval[2]) < p[i])
			while (h(interval[1], mu,  sigma, nu, tau, true, false) < temp) {
				//interval[2]<- mu[i]+j*sigma[i]
				interval[1] = mu + j * sigma;
				
				//j<-j+1
				j++;
			}
		} else {
			
			//interval <-  c(mu[i]-sigma[i], mu[i])
			interval[0] = mu - sigma;
			interval[1] = mu;
			
			//j <-2
			int j = 2;
			
	        //while (h(interval[1]) > p[i])
			while (h(interval[0], mu,  sigma, nu, tau, true, false) > temp) {
				
	            //interval[1]<- mu[i]-j*sigma[i]
				interval[0] = mu - j * sigma;
				
				//j<-j+1 
				j++;     
			}
		}
		
		//q[i] <- uniroot(h1, interval)$root
		uniRootObj.setParameters(out, mu, sigma, nu, tau, temp);
		if (interval[0] < interval[1]) {
			out = uniRootSolver.solve(1000, uniRootObj, interval[0], interval[1]);
		} else {
			out = uniRootSolver.solve(1000, uniRootObj, interval[1], interval[0]);
		}
		return out;
	}
	
	/**
	 * Supportive function for qST1, calss pST1.
	 * @param in - income value
	* @param mu -  vector of mu distribution parameters values
	* @param sigma -  vector of sigma distribution parameters values
	* @param nu -  vector of nu distribution parameters values
	* @param tau -  vector of tau distribution parameters values
	* @param lowerTail - logical; if TRUE (default), probabilities 
	* are P[X <= x] otherwise, P[X > x]
	* @param isLog - logical; if TRUE, probabilities p are given as log(p).
	 * @return value of cumulative probability function
	 */
	public final double h(final double in, 
                          final double  mu, 
                          final double  sigma, 
                          final double nu, 
                          final double tau, 
                          final boolean lowerTail, 
                          final boolean isLog) {
		
		return pST1(in, mu, sigma, nu, tau, lowerTail, isLog);
	}
	
	/**
  	 * Inner class is a shell for objective function to
  	 * find the root of the function.
  	 *
  	 */
	class UniRootObjFunction implements UnivariateFunction {
		
		 /** mu distribution parameter. */  
		private double mu;
		/** sigma distribution parameter. */  
		private double sigma;	
		 /** nu distribution parameter. */  
		private double nu;
		/** tau distribution parameter. */  
		private double tau;		
		/** income value. */  
		private double q;
		/** value of cumulative probability. */  
		private double p;
				
		/**
		 * This function is used to integrate the objectivefunction function.
		 * @param x - income value to determine zero of the function
		 * @return value of the function
		 */
		public double value(final double x) {
			return h1(q, mu, sigma, nu, tau, true, false, p);
		}	
		
		/**
		 * Set required paeameters.
		 * @param q - income value		 
		 * @param mu -  value of mu distribution parameters
		 * @param sigma -  value of sigma distribution parameters
		 * @param nu -  value of nu distribution parameters 
		 * @param tau -  value of tau distribution parameters
		 * @param p - value of cumulative probability
		 */
	    public void setParameters(final double q, 
                				  final double  mu, 
                				  final double  sigma, 
                				  final double nu, 
                				  final double tau,
                				  final double p) {
	    	this.mu = mu;
	    	this.sigma = sigma;
	    	this.nu = nu;
	    	this.tau = tau;
	    	this.q = q;
			this.p = p;
		}	
	}
	
	/**
	 * Supportive function for qGTs, calss pGT.
	 * @param in - income value
	* @param mu -  vector of mu distribution parameters values
	* @param sigma -  vector of sigma distribution parameters values
	* @param nu -  vector of nu distribution parameters values
	* @param tau -  vector of tau distribution parameters values
	* @param lowerTail - logical; if TRUE (default), probabilities 
	* are P[X <= x] otherwise, P[X > x]
	* @param isLog - logical; if TRUE, probabilities p are given as log(p).
	* @param p - value of cumulative probability
	 * @return value of cumulative probability function
	 */
	public final double h1(final double in,
                          final double  mu, 
                          final double  sigma, 
                          final double nu, 
                          final double tau, 
                          final boolean lowerTail, 
                          final boolean isLog,
                          final double p) {
		
		return (pST1(in, mu, sigma, nu, tau, lowerTail, isLog) - p);
	}
	
	/**
	 * qST1(p) launches qST1(p, mu, sigma, nu,  tau, lowerTail, isLog)
	 *  with deafult mmu = 0, sigma = 1, nu = 0, tau = 2.
	 * lowerTail = true, isLog = false.
	 * @param p - value of cumulative probability 
	 * @return quantile
	 */
	//qST1 <-  function(p, mu = 0, sigma = 1, nu = 0, 
	//tau = 2, lower.tail = TRUE, log.p = FALSE)
	public final double qST1(final double p) {
		return qST1(p, 0.0, 1.0, 0.0, 2.0, true, false);
	}
	
	/** Generates a random sample from this distribution.
	 * @param mu -  vector of mu distribution parameters values
	 * @param sigma -  vector of sigma distribution parameters values
	 * @param nu -  vector of nu distribution parameters values
	 * @param tau -  vector of tau distribution parameters values
	 * @param uDist -  object of UniformRealDistribution class;
	 * @return random sample vector
	 */	
		public final double rST1(final double mu, 
		       	   				 final double sigma, 
		       	   				 final double nu, 
		       	   				 final double tau,
		       	   				 final UniformRealDistribution uDist) {	
			
			// {if (any(sigma <= 0))stop(paste("sigma must be positive"))
			if (sigma <= 0) {
				System.err.println("sigma must be positive");
				return -1.0;
			}			    
			//r <- qST1(p,mu=mu,sigma=sigma,nu=nu,tau=tau)
			return qST1(uDist.sample(), mu, sigma, nu, tau, true, false);
		}
	
	/**
	* rST1(uDist) launches rST1(mu, sigma, nu,  tau, uDist) 
	* with deafult mu=0, sigma=1, nu=0, tau=2.
		* @param uDist -  object of UniformRealDistribution class;
	 * @return random sample value
	*/
	//rST1 <- function(n, mu=0, sigma=1, nu=0, tau=2)
	public final double  rST1(final UniformRealDistribution uDist) {
		return rST1(0.0, 1.0, 0.0, 2.0, uDist);
	}	
	
	/**
	* Checks whether the mu distribution parameter is valid.
	* @param y - vector of response variavbles
	* @return - boolean
	*/
	public final boolean isYvalid(final ArrayRealVector y) {
		return true;
	}

	/** Checks whether entries of ArrayRealVectors 
	 * of distribution parameters are valid.
	* @param whichDistParameter - distribution parameter
	  @return Hashtable of booleans
	 */
	public final boolean areDistributionParametersValid(
											final int whichDistParameter) {
		boolean tempB = false;
		switch (whichDistParameter) {
      case DistributionSettings.MU:
      	tempB = isMuValid(
      				distributionParameters.get(DistributionSettings.MU));
         break;
      case DistributionSettings.SIGMA:
      	tempB = isSigmaValid(
      				distributionParameters.get(DistributionSettings.SIGMA));
        break;
      case DistributionSettings.NU:
      	tempB = isNuValid(
      				distributionParameters.get(DistributionSettings.NU));
        break;
      case DistributionSettings.TAU:
      	tempB = isTauValid(
      				distributionParameters.get(DistributionSettings.TAU));
        break;
		default: System.out.println("The specific distribution parameter"
				+ " does not exist for this distribution");
		}
		return tempB;
	}
	
	
	/**
	* Checks whether the mu distribution parameter is valid.
	* @param mu - vector of mu (mean) values
	* @return - boolean
	*/
	private boolean isMuValid(final ArrayRealVector mu) {
		//mu.valid = function(mu) TRUE,
		return true;
	}
	
	/**
	 * Checks whether the sigma distribution parameter is valid.
	 * @param sigma - vector of sigma (standard deviation) values
	 * @return - - boolean
	 */
	private boolean isSigmaValid(final ArrayRealVector sigma) {
		return sigma.getMinValue() > 0;	
	}	

	/**
	 * Checks whether the nu distribution parameter is valid.
	 * @param nu - vector of nu values
	 * @return - - boolean
	 */
	private boolean isNuValid(final ArrayRealVector nu) {
		//nu.valid = function(nu) TRUE ,
			return true;	
	}

	/**
	 * Checks whether the tau distribution parameter is valid.
	 * @param tau - vector of nu values
	 * @return - - boolean
	 */
	private boolean isTauValid(final ArrayRealVector tau) {
		return tau.getMinValue() > 0;	
	}
	
	/**
	 * Get number of distribution parameters.
	 * @return number of distribution parameters
	 */
	public final int getNumberOfDistribtionParameters() {
		return numDistPar;
	}

	/**
	 * Get type of distributuion (Continuous, Discrete or Mixed).
	 * @return type of distributuion
	 */
	public final int getTypeOfDistribution() {
		return DistributionSettings.CONTINUOUS;
	}

	/**
	 * Get distribution name.
	 * @return distribution name.
	 */
	public final int getFamilyOfDistribution() {
		return DistributionSettings.ST1;
	}


	/**
	 * Set distribution parsameters.
	 * @param whichDistParameter - the fitting distribution parameter
	 * @param fvDistributionParameter - vector of values of 
	 * fitting distribution parameter
	 */
	public final void setDistributionParameter(
							final int whichDistParameter, 
							final ArrayRealVector fvDistributionParameter) {
		this.distributionParameters.put(whichDistParameter, 
												fvDistributionParameter);
	}

	/**
	 * Get distribution parsameters.
	 * @param whichDistParameter - distribution parameter
	 * @return - vector of distribution parameter values
	 */
	public final ArrayRealVector getDistributionParameter(
												final int whichDistParameter) {
		return this.distributionParameters.get(whichDistParameter);
	}

	/** Get the link function type of the current distribution parameter.
	 * @param whichDistParameter - distribution parameter
	 * @return link function type
	 */
	public final int getDistributionParameterLink(
												final int whichDistParameter) {
		return distributionParameterLink.get(whichDistParameter);
	}
}
