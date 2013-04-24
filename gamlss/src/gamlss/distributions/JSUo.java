/*
  Copyright 2012 by Dr. Vlasios Voudouris and ABM Analytics Ltd
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/
package gamlss.distributions;

import gamlss.utilities.Controls;
import gamlss.utilities.MakeLinkFunction;
import gamlss.utilities.NormalDistr;

import java.util.Hashtable;

import org.apache.commons.math3.distribution.TDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.util.FastMath;

/**
 * @author Dr. Vlasios Voudouris, Daniil Kiose, 
 * Prof. Mikis Stasinopoulos and Dr Robert Rigby.
 */
public class JSUo implements GAMLSSFamilyDistribution {
	
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
	private double[] r;
	/** Temporary int for interim operations. */
	private int size;
	/** Object of the Normal distribution.  */
	private NormalDistr noDist;
	
	/** This is the Johnson SU original  distribution with 
	 * default link (muLink="identity",sigmaLink="log", 
	 * nuLink="identity", tauLink="log"). */
	public JSUo() {
			
		this(DistributionSettings.IDENTITY, 
				 DistributionSettings.LOG, 
				 DistributionSettings.IDENTITY, 
				 DistributionSettings.LOG);
	}
			
	/**
	 * This is the the Johnson SU original distribution with supplied link 
	 * function for each of the distribution parameters.
	 * @param muLink - link function for mu distribution parameter
	 * @param sigmaLink - link function for sigma distribution parameter
	 * @param nuLink - link function for nu distribution parameter
	 * @param tauLink - link function for tau distribution parameter
	 */
	public JSUo(final int muLink, 
			   final int sigmaLink, 
			   final int nuLink, 
			   final int tauLink) {
				
		distributionParameterLink.put(DistributionSettings.MU,
				 MakeLinkFunction.checkLink(DistributionSettings.JSUo, muLink));
		distributionParameterLink.put(DistributionSettings.SIGMA,
			  MakeLinkFunction.checkLink(DistributionSettings.JSUo, sigmaLink));
		distributionParameterLink.put(DistributionSettings.NU,
				 MakeLinkFunction.checkLink(DistributionSettings.JSUo, nuLink));
		distributionParameterLink.put(DistributionSettings.TAU,
				MakeLinkFunction.checkLink(DistributionSettings.JSUo, tauLink));
		
		//p <- pNO(r,0,1)
		noDist = new NormalDistr(0.0, 1.0);
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
		//sigma.initial = expression(sigma<- rep(.1, length(y))),
		tempV = new ArrayRealVector(y.getDimension());
		tempV.set(0.1);
		return tempV;
	}

	
	
	/** Calculate and set initial value of nu.
	 * @param y - vector of values of response variable
	 * @return vector of initial values of nu
	 */
	private ArrayRealVector setNuInitial(final ArrayRealVector y) {	
		//nu.initial = expression(nu <- rep(0, length(y))),
		return new ArrayRealVector(y.getDimension());
	}

	

	/** Calculates initial value of tau.
	 * @param y - vector of values of response variable
	 * @return vector of initial values of tau
	 */
	private ArrayRealVector setTauInitial(final ArrayRealVector y) {
		//tau.initial = expression(tau <-rep(0.5, length(y))),
		tempV = new ArrayRealVector(y.getDimension());
		tempV.set(0.5);
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
		setInterimArrays(y);
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
	 * Set z and r arrays.
	 * @param y - response variable
	 */
	private void setInterimArrays(final ArrayRealVector y) {
		muV     = distributionParameters.get(DistributionSettings.MU);
		sigmaV  = distributionParameters.get(DistributionSettings.SIGMA);
	 	nuV     = distributionParameters.get(DistributionSettings.NU);
	 	tauV    = distributionParameters.get(DistributionSettings.TAU);
		
	 	size = y.getDimension();
		z = new double[size];
		r = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		
	 		//z <- (y-mu)/sigma
	 		z[i] = (y.getEntry(i) - muV.getEntry(i)) / sigmaV.getEntry(i);
	 		
	 	    //r <- nu + tau*asinh(z)
	 		r[i] = nuV.getEntry(i) + tauV.getEntry(i) * FastMath.asinh(z[i]);
	 	}
	}

	/**  First derivative dldm = dl/dmu, where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of first derivative dldm = dl/dmu
	 */	
	public final ArrayRealVector dldm(final ArrayRealVector y) { 

	 	dldm = new double[size];
	 	for (int i = 0; i < size; i++) {
		
		//dldm <- (z/(sigma*(z*z+1)))+((r*tau)/(sigma*(z*z+1)^(0.5)))
	 	dldm[i] = (z[i] / (sigmaV.getEntry(i) * (z[i] 
	 			* z[i] + 1))) + ((r[i] * tauV.getEntry(i)) 
	 					/ (sigmaV.getEntry(i) * FastMath.sqrt(z[i] 
	 													* z[i] + 1)));
	 	}
	 	r = null;
	 	z = null;
	 	return new ArrayRealVector(dldm, false);
	}
	
	/** First derivative dlds = dl/dsigma, where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of First derivative dlds = dl/dsigma
	 */	
	public final ArrayRealVector dlds(final ArrayRealVector y) {	
		
		dlds = new double[size];
	 	for (int i = 0; i < size; i++) {	
	 		//dldd <- (-1/(sigma*(z*z+1)))+((r*tau*z)/(sigma*(z*z+1)^(0.5)))
	 		dlds[i] = (-1 / (sigmaV.getEntry(i) * (z[i] 
	 				* z[i] + 1))) + ((r[i] * tauV.getEntry(i) 
	 						* z[i]) / (sigmaV.getEntry(i) 
	 								* FastMath.sqrt(z[i] * z[i] + 1)));
	 	}
	 	r = null;
	 	z = null;
	 	return new ArrayRealVector(dlds, false);
	}

	/** First derivative dldn = dl/dnu, where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of First derivative dldn = dl/dnu
	 */	
	public final ArrayRealVector dldn(final ArrayRealVector y) {	

		dldn = new double[size];
	 	for (int i = 0; i < size; i++) {

			//dldv <- -r
	 		dldn[i] = -r[i];
		 }
	 	r = null;
	 	z = null;
		return new ArrayRealVector(dldn, false);
	}
	
	/** First derivative dldtau = dl/dtau, where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of First derivative dldtau = dl/dtau
	 */
	public final ArrayRealVector dldt(final ArrayRealVector y) {	
		
		dldt = new double[size];
	 	for (int i = 0; i < size; i++) {
			
			//dldt <- (1+r*nu-r*r)/tau
			dldt[i] = (1 + r[i] * nuV.getEntry(i) - r[i] * r[i]) 
													/ tauV.getEntry(i);
		 }
	 	 r = null;
	 	 z = null;
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
			
			out[i] = (-2) * dJSUo(y.getEntry(i), muArr[i], sigmaArr[i], 
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
		public final double dJSUo(final double x, 
				                  final double mu, 
				                  final double sigma, 
				                  final double nu, 
				                  final double tau, 
				                  final boolean isLog) {
			
			// {  if (any(sigma <= 0))stop(paste("sigma must be positive",))
			if (sigma < 0) {
				System.err.println("sigma must be positive");
				return -1.0;
			}
			//if (any(tau < 0))  stop(paste("tau must be positive", "\n", ""))
			if (tau < 0) {
				System.err.println("nu must be positive");
				return -1.0;
			}
			
		    //z <- (x-mu)/sigma
			final double z = (x - mu) / sigma;
			
			//r <- nu + tau*asinh(z)
			final double r = nu + tau * FastMath.asinh(z);
			
			//loglik <- -log(sigma)+ log(tau)- 
			//.5*log(z*z+1) -.5*log(2*pi)-.5*r*r
			double out = -FastMath.log(sigma) + FastMath.log(tau) 
								- 0.5 * FastMath.log(z * z  + 1) - 0.5  
							* FastMath.log(2 * FastMath.PI) - 0.5 * r * r;
					
			//if(log==FALSE) ft  <- exp(loglik) else ft <- loglik
			if (!isLog) {
				out = FastMath.exp(out);
			}
		 	return out;
		}

	/**
	 * dJSUo(x) launches dJSUo(x, mu, sigma, nu, isLog) 
	 * with deafult mu=0, sigma=1, nu=0, tau = 1, isLof=false.
	 * @param x - value of response variable  
	 * @return value of probability density function  
	 */
	//dJSUo <- function(x, mu=0, sigma=1, nu=0, tau=1, log=FALSE)
	public final double dJSUo(final double x) {
		return dJSUo(x, 0.0, 1.0, 0.0, 1.0, false);
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
	public final double pJSUo(final double q, 
 	   		                  final double mu, 
 	   		                  final double sigma, 
 	   		                  final double nu,
 	   		                  final double tau, 
 	   		                  final boolean lowerTail, 
 	   		                  final boolean isLog) {

		// {  if (any(sigma <= 0))stop(paste("sigma must be positive",))
		if (sigma < 0) {
			System.err.println("sigma must be positive");
			return -1.0;
		}
		//if (any(tau < 0))  stop(paste("tau must be positive", "\n", ""))
		if (tau < 0) {
			System.err.println("nu must be positive");
			return -1.0;
		}

	    //z <- (x-mu)/sigma
		//r <- nu + tau*asinh(z)
		final double  r = nu + tau * FastMath.asinh((q - mu) / sigma);
		
		//p <- pNO(r,0,1)
		double out = noDist.cumulativeProbability(r);
		
		//if(lower.tail==TRUE) p  <- p else  p <- 1-p
		//if(log.p==FALSE) p  <- p else  p <- log(p)
	    if (!lowerTail) {
	    	if (isLog) {
	    		out = FastMath.log(1 - out);
	    	} else {
	    		out = 1 - out;
	    	}
	    } else if (isLog) {
	    	out = FastMath.log(out);
	    }
	    return out;
	}
	 		
	/**
	 * pJSUo(q) launches pJSUo(q, mu, sigma, nu,  tau, lowerTail, isLog)
	 *  with deafult mu=0, sigma=1, nu=0, tau = 1.
	 * lowerTail = true, isLog = false.
	 * @param q - quantile
	 * @return value of cumulative probability function P(X <= q)
	 */
	public final double  pJSUo(final double q) {
		//pJSUo <- function(q, mu=0, sigma=1, nu=0, 
		//tau=1, lower.tail = TRUE, log.p = FALSE)
		return pJSUo(q, 0.0, 1.0, 0.0, 1.0, true, false);
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
	public final double qJSUo(final double p, 
	                          final double  mu, 
	                          final double  sigma, 
	                          final double nu, 
	                          final double tau, 
	                          final boolean lowerTail, 
	                          final boolean isLog) {
		
		// {  if (any(sigma <= 0))stop(paste("sigma must be positive",))
		if (sigma < 0) {
			System.err.println("sigma must be positive");
			return -1.0;
		}
		//if (any(tau < 0))  stop(paste("tau must be positive", "\n", ""))
		if (tau < 0) {
			System.err.println("nu must be positive");
			return -1.0;
		}
	
		double out = p;
		//if (log.p==TRUE) p <- exp(p) else p <- p
		if (isLog) {
			out = FastMath.exp(out);
		}
		
		//if (any(p <= 0)|any(p >= 1))  
		//stop(paste("p must be between 0 and 1", "\n", "")) 
		if (out <= 0 || out >= 1) {
			System.err.println("p must be between 0 and 1");
		}
		
		//if (lower.tail==TRUE) p <- p else p <- 1-p
		if (!lowerTail) {
			out = 1 - out;
		}
	    
		//r <- qNO(p,0,1)
	    //z <- sinh((r-nu)/tau)     
	    final double z = FastMath.sinh((
	    				noDist.inverseCumulativeProbability(out) - nu) / tau);
		
		//q <- mu+sigma*z   
		out = mu + sigma * z;
	
		return out;
	}

	/**
	 * qJSUo(p) launches qJSUo(p, mu, sigma, nu,  tau, lowerTail, isLog)
	 *  with deafult mu=0, sigma=1, nu=0, tau=1.
	 * lowerTail = true, isLog = false.
	 * @param p - value of cumulative probability
	 * @return value of quantile function
	 */
	//qJSUo <-  function(p, mu=0, sigma=1, nu=0, tau=1, lower.tail = TRUE, log.p = FALSE)
	public final double qJSUo(final double p) {

		return qJSUo(p, 0.0, 1.0, 0.0, 1.0, true, false);
	}	

	/** Generates a random sample from this distribution.
	 * @param mu -  vector of mu distribution parameters values
	 * @param sigma -  vector of sigma distribution parameters values
	 * @param nu -  vector of nu distribution parameters values
	 * @param tau -  vector of tau distribution parameters values
	 * @param uDist -  object of UniformRealDistribution class;
	 * @return random sample vector
	 */	
		public final double rJSUo(final double mu, 
		       	                  final double sigma, 
		       	                  final double nu, 
		       	                  final double tau,
		       	                  final UniformRealDistribution uDist) {
			
			// {  if (any(sigma <= 0))stop(paste("sigma must be positive",))
			if (sigma <= 0) {
				System.err.println("sigma must be positive");
				return -1.0;
			}
	
			//n <- ceiling(n)
			//temp = FastMath.ceil(n.getEntry(i));	    
			//r <- qST3(p,mu=mu,sigma=sigma,nu=nu,tau=tau)
			return qJSUo(uDist.sample(), mu, sigma, nu, tau, true, false);
		}
	
	/**
	* rJSUo(n) launches rJSUo(n, mu, sigma, nu,  tau)
	*  with deafult mu=0, sigma=1, nu=0, tau=1.
	* @param uDist -  object of UniformRealDistribution class;
	* @return random sample value
	*/
	//rJSUo <- function(n, mu=0, sigma=1, nu=0, tau=1)
	public final double rJSUo(final UniformRealDistribution uDist) {
		return rJSUo(0.0, 1.0, 0.0, 1.0, uDist);
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
		return DistributionSettings.JSUo;
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
