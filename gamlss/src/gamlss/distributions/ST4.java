/*
  Copyright 2012 by Dr. Vlasios Voudouris and ABM Analytics Ltd
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/
package gamlss.distributions;

import gamlss.utilities.Controls;
import gamlss.utilities.MakeLinkFunction;
import gamlss.utilities.TDistr;

import java.util.Hashtable;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.util.FastMath;

/**
 * @author Dr. Vlasios Voudouris, Daniil Kiose, 
 * Prof. Mikis Stasinopoulos and Dr Robert Rigby.
 */
public class ST4 implements GAMLSSFamilyDistribution {

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
	/** Object of t distribution class. */
	private TDistr tdDist;
	/** Temporary array for interim operations. */
	private double[] dsq;
	/** Temporary array for interim operations. */
	private double[] w;
	/** Temporary int for interim operations. */
	private int size;
	/** Temporary vector for interim operations. */
	private ArrayRealVector tempV;
	/** Array of first derrivative values dl/dmu. */
	private double[] dldm;
	/** Array of first derrivative values dl/dsigma. */
	private double[] dlds;
	/** Array of first derrivative values dl/dnu. */
	private double[] dldn;
	/** Array of first derrivative values dl/dtau. */
	private double[] dldt;
	/** Object of Normal distribution class. */
	private NormalDistribution noDist;
	
	/** This is the Skew t type4  distribution with default link 
	 * (muLink="identity",sigmaLink="log", nuLink="log", tauLink="log"). */
	public ST4() {
			
		this(DistributionSettings.IDENTITY, 
			 DistributionSettings.LOG, 
			 DistributionSettings.LOG, 
			 DistributionSettings.LOG);
	}
			
	/** This is the Skew t type4 distribution with supplied 
	 * link function for each of the distribution parameters.
	 * @param muLink - link function for mu distribution parameter
	 * @param sigmaLink - link function for sigma distribution parameter
	 * @param nuLink - link function for nu distribution parameter
	 * @param tauLink - link function for tau distribution parameter
	 */
	public ST4(final int muLink, 
			   final int sigmaLink, 
			   final int nuLink, 
			   final int tauLink) {
				
		distributionParameterLink.put(DistributionSettings.MU, 	   
			MakeLinkFunction.checkLink(DistributionSettings.ST4, muLink));
		distributionParameterLink.put(DistributionSettings.SIGMA,  
			MakeLinkFunction.checkLink(DistributionSettings.ST4, sigmaLink));
		distributionParameterLink.put(DistributionSettings.NU,     
			MakeLinkFunction.checkLink(DistributionSettings.ST4, nuLink));
		distributionParameterLink.put(DistributionSettings.TAU,    
			MakeLinkFunction.checkLink(DistributionSettings.ST4, tauLink));
		
		noDist = new NormalDistribution();
		tdDist = new TDistr();
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
		//sigma.initial = expression(sigma<- rep(sd(y)/4, length(y))),
		tempV = new ArrayRealVector(y.getDimension());
		final double out = new StandardDeviation().evaluate(y.getDataRef());
		tempV.set(out / 4.0);
		return tempV;
	}

	/** Calculate and set initial value of nu.
	 * @param y - vector of values of response variable
	 * @return vector of initial values of nu
	 */
	private ArrayRealVector setNuInitial(final ArrayRealVector y) {	
		//nu.initial = expression(nu <- rep(5, length(y))),
		tempV = new ArrayRealVector(y.getDimension());
		tempV.set(5.0);
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
	 * Set s, dsq, w, ym arrays.
	 * @param y - response variable
	 */
	private void setInterimArrays(final ArrayRealVector y) {
		muV     = distributionParameters.get(DistributionSettings.MU);
		sigmaV  = distributionParameters.get(DistributionSettings.SIGMA);
	 	nuV     = distributionParameters.get(DistributionSettings.NU);
	 	tauV    = distributionParameters.get(DistributionSettings.TAU);
		
	 	size = y.getDimension();
		dsq = new double[size];
		w = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		
	 		//dsq  <- ((y-mu)/s)^2
	 		dsq[i] = FastMath.pow(((y.getEntry(i) - muV.getEntry(i)) 
	 												/ sigmaV.getEntry(i)), 2);
	 			
	 		if (y.getEntry(i) < muV.getEntry(i)) {
	 			
		 		//w1 <- ifelse(nu < 1000000, (nu+1)/(nu+dsq),1)	
		 		if (nuV.getEntry(i) < 1000000) {
		 			w[i] = (nuV.getEntry(i) + 1) 
		 							/ (nuV.getEntry(i) + dsq[i]);
		 		} else {
		 			
		 			w[i] = 1.0;
		 		}
	 		} else {
	 			
	 			//w2 <- ifelse(tau < 1000000, (tau+1)/(tau+dsq),1)
	 			if (tauV.getEntry(i) < 1000000){
	 				
	 				w[i] = (tauV.getEntry(i) + 1) 
	 								/ (tauV.getEntry(i) + dsq[i]);
	 			} else {
	 				
	 				w[i] = 1.0;
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

 		//dldm <- ifelse(y < mu, (w1*(y-mu))/(s^2) , (w2*(y-mu))/(s^2))
	 	dldm[i] = (w[i] * (y.getEntry(i) - muV.getEntry(i))) 
	 						/ (sigmaV.getEntry(i) * sigmaV.getEntry(i));
	 	}
	 	dsq = null;
	 	w = null;
	 	return new ArrayRealVector(dldm, false);
	}
	
	

	/** First derivative dlds = dl/dsigma, where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of First derivative dlds = dl/dsigma
	 */	
	public final ArrayRealVector dlds(final ArrayRealVector y) {	
		
		dlds = new double[size];
	 	for (int i = 0; i < size; i++) {

	 		//dldd <- ifelse(y < mu, (w1*dsq-1)
	 		///(sigma) , (w2*dsq-1)/(sigma))
	 		dlds[i] = (w[i] * dsq[i] - 1) / (sigmaV.getEntry(i));

	 	}
	 	dsq = null;
	 	w = null;
	 	return new ArrayRealVector(dlds, false);
}
	
	/** First derivative dldn = dl/dnu, where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of First derivative dldn = dl/dnu
	 */	
	public final ArrayRealVector dldn(final ArrayRealVector y) {	

		dldn = new double[size];
		double lk1 = 0;
		double lk2 = 0;
	 	for (int i = 0; i < size; i++) {

			if (nuV.getEntry(i) < 1000000) {
				
				//lk1 <- lgamma(0.5) + lgamma(nu/2) 
				//- lgamma((nu+1)/2) + 0.5*log(nu
				lk1 = Gamma.logGamma(0.5) 
						+ Gamma.logGamma(nuV.getEntry(i) / 2) 
							- Gamma.logGamma((nuV.getEntry(i) + 1) 
								/ 2) + 0.5 * FastMath.log(nuV.getEntry(i));
			} else {
				
				lk1 = 0.5 * FastMath.log(2.0 * FastMath.PI);
			}

			//lk2 <- ifelse(tau < 1000000, lk2, 0.5*log(2*pi))
			if (tauV.getEntry(i) < 1000000) {
				
				//lk2 <- lgamma(0.5) + lgamma(tau/2) 
				//- lgamma((tau+1)/2) + 0.5*log(tau)
				lk2 = Gamma.logGamma(0.5) 
						+ Gamma.logGamma(tauV.getEntry(i) / 2) 
						- Gamma.logGamma((tauV.getEntry(i) + 1) / 2) 
									+ 0.5 * FastMath.log(tauV.getEntry(i));	
			} else {
				
				lk2 = 0.5 * FastMath.log(2.0 * FastMath.PI);
			}
			
			//lk <- lk2 - lk1			
			//k <- exp(lk)
			final double k = FastMath.exp(lk2 - lk1);
			
			//dldva <- -0.5*log(1+dsq/nu)+(w1*dsq)/(2*nu)
			//dldv <- ifelse(y < mu, dldva , 0)
			if (y.getEntry(i) < muV.getEntry(i)) {
				dldn[i] = -0.5 * FastMath.log(1 + dsq[i] 
						/ nuV.getEntry(i)) + (w[i] * dsq[i]) 
										/ (2.0 * nuV.getEntry(i));
			} else {
				
				dldn[i] = 0.0;
			}
			
			//dldv <- dldv+0.5*(digamma((nu+1)/2)
			//-digamma(nu/2)-(1/nu))*(1/(1+k))
			dldn[i] = dldn[i] + 0.5 * (Gamma.digamma((nuV.getEntry(i)
						+ 1) / 2.0) - Gamma.digamma(nuV.getEntry(i) / 2.0)
									- (1 / nuV.getEntry(i))) * (1.0 / (1 + k));
		 }
	 	dsq = null;
	 	w = null;
	 	return new ArrayRealVector(dldn, false);
	}

	/** First derivative dldtau = dl/dtau, where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of First derivative dldtau = dl/dtau
	 */
	public final ArrayRealVector dldt(final ArrayRealVector y) {	
		
		dldt = new double[size];
		double lk1 = 0;
		double lk2 = 0;
	 	for (int i = 0; i < size; i++) {

	 		//lk1 <- lgamma(0.5) + lgamma(nu/2) 
	 		//- lgamma((nu+1)/2) + 0.5*log(nu)
	 		//lk1 <- ifelse(nu < 1000000, lk1, 0.5*log(2*pi))
	 		if (nuV.getEntry(i) < 1000000) {
	 			
				lk1 = Gamma.logGamma(0.5) 
						+ Gamma.logGamma(nuV.getEntry(i) / 2) 
							- Gamma.logGamma((nuV.getEntry(i) + 1) 
								/ 2) + 0.5 * FastMath.log(nuV.getEntry(i));
	 		} else {
	 			
	 			lk1 = 0.5 * FastMath.log(2.0 * FastMath.PI);
	 		}
	 		
	 		//lk2 <- ifelse(tau < 1000000, lk2, 0.5*log(2*pi))
			if (tauV.getEntry(i) < 1000000) {
				
				//lk2 <- lgamma(0.5) + lgamma(tau/2) 
				//- lgamma((tau+1)/2) + 0.5*log(tau)
				lk2 = Gamma.logGamma(0.5) 
						+ Gamma.logGamma(tauV.getEntry(i) / 2) 
						- Gamma.logGamma((tauV.getEntry(i) + 1) / 2) 
									+ 0.5 * FastMath.log(tauV.getEntry(i));	
			} else {
				
				lk2 = 0.5 * FastMath.log(2.0 * FastMath.PI);
			}
	 		
			//lk <- lk2 - lk1			
			//k <- exp(lk)
			final double k = FastMath.exp(lk2 - lk1);
	 		
			if (y.getEntry(i) < muV.getEntry(i)) {

				dldt[i] = 0.0;
			} else {
				
				//-0.5*log(1+dsq/tau)+(w2*dsq)/(2*tau)
				dldt[i] = -0.5 * FastMath.log(1 + dsq[i] 
						/ tauV.getEntry(i)) + (w[i] * dsq[i]) 
										/ (2.0 * tauV.getEntry(i));
			}
	 		
			//dldt <- dldt+0.5*(digamma((tau+1)/2)
			//-digamma(tau/2)-(1/tau))*(k/(1+k))
			dldt[i] = dldt[i] + 0.5 * (Gamma.digamma((tauV.getEntry(i) 
					+ 1) / 2.0) - Gamma.digamma(tauV.getEntry(i) / 2.0) 
								- (1 / tauV.getEntry(i))) * (k / (1 + k));
	 	}
	 	dsq = null;
	 	w = null;
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
			
			out[i] = (-2) * dST4(y.getEntry(i), muArr[i], sigmaArr[i], 
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
	public final double dST4(final double x, 
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
		
		//if (any(nu <= 0))  stop(paste("nu must be positive", "\n", ""))
		if (nu <= 0) {
			System.err.println("nu must be positive");
			return -1.0;
		}
		
		//if (any(tau <= 0))  stop(paste("tau must be positive", "\n", ""))
		if (tau <= 0) {
			System.err.println("tau must be positive");
			return -1.0;
		}
			
		double lk1 = 0;
		double lk2 = 0;
		//lk1 <- lgamma(0.5) + lgamma(nu/2) 
 		//- lgamma((nu+1)/2) + 0.5*log(nu)
 		//lk1 <- ifelse(nu < 1000000, lk1, 0.5*log(2*pi))
 		if (nu < 1000000) {
 			
			lk1 = Gamma.logGamma(0.5) 
					+ Gamma.logGamma(nu / 2) 
						- Gamma.logGamma((nu + 1) 
							/ 2) + 0.5 * FastMath.log(nu);
 		} else {
 			
 			lk1 = 0.5 * FastMath.log(2.0 * FastMath.PI);
 		}
 		
 		//lk2 <- ifelse(tau < 1000000, lk2, 0.5*log(2*pi))
		if (tau < 1000000) {
			
			//lk2 <- lgamma(0.5) + lgamma(tau/2) 
			//- lgamma((tau+1)/2) + 0.5*log(tau)
			lk2 = Gamma.logGamma(0.5) 
					+ Gamma.logGamma(tau / 2) 
					- Gamma.logGamma((tau + 1) / 2) 
								+ 0.5 * FastMath.log(tau);	
		} else {
			
			lk2 = 0.5 * FastMath.log(2.0 * FastMath.PI);
		}
 		
		//lk <- lk2 - lk1	
		final double lk = lk2 - lk1;
		//k <- exp(lk)
		final double k = FastMath.exp(lk);
		double out = 0;
				
		//loglik <- ifelse(x < mu, loglik1, loglik2 + lk)
		if (x < mu) { 
			
			//loglik1 <- if (tau<1000000) loglik1 else  loglikb
			if (tau < 1000000) {
				
				//loglik1 <- dt((x-mu)/sigma, df=nu, log=TRUE)
				tdDist.setDegreesOfFreedom(nu);
				out = FastMath.log(tdDist.density((x-mu)/sigma));
			} else {
				
				//loglikb <- dNO(((x-mu)/sigma), mu=0, sigma=1, log=TRUE)
				out =  FastMath.log(noDist.density((x - mu) / sigma));
			}
		} else {
			
		    //loglik2 <- ifelse(tau<1000000,loglik2,loglikb)
			if(tau < 1000000)
			{
				//loglik2 <- dt((x-mu)/sigma, df=tau, log=TRUE)
				tdDist.setDegreesOfFreedom(tau);
				out = FastMath.log(tdDist.density((x - mu) / sigma)) + lk;
			} else {
				
				//loglikb <- dNO(((x-mu)/sigma), mu=0, sigma=1, log=TRUE)
				out = FastMath.log(noDist.density(((x - mu) / sigma))) + lk;
			}
		}
		
		//loglik <- loglik + log(2/(1+k)) - log(sigma)
		out = out + FastMath.log(2.0 / (1 + k)) - FastMath.log(sigma);
		
		//if(log==FALSE) ft  <- exp(loglik) else ft <- loglik
		if(!isLog)
		{
			out = FastMath.exp(out);
		}
		
		return out;
	}
	
	/**
	 * dST4(x) launches dST4(x, mu, sigma, nu, isLog) 
	 * with deafult mu=0, sigma=1, nu=1, tau = 10, isLog=false.
	 * @param x - value of response variable 
	 * @return value of probability density function
	 */
	public final double dST4(final double  x) {
		//dST4 <- (x, mu=0, sigma=1, nu=1, tau=10, log=FALSE)
		return dST4(x, 0.0, 1.0, 1.0, 10.0, false);
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
	public final double pST4(final double q, 
 	   		 				  final double mu, 
 	   		 				  final double sigma, 
 	   		 				  final double nu,
 	   		 				  final double tau, 
 	   		 				  final boolean lowerTail, 
 	   		 				  final boolean isLog) {
	
		// {  if (any(sigma < 0))stop(paste("sigma must be positive",))
		if (sigma < 0) {
			System.err.println("sigma must be positive");
			return -1.0;
		}
		
		//if (any(nu <= 0))  stop(paste("nu must be positive", "\n", ""))
		if (nu <= 0) {
			System.err.println("nu must be positive");
			return -1.0;
		}
		
		//if (any(tau < 0))  stop(paste("tau must be positive", "\n", ""))
		if (tau < 0) {
			System.err.println("tau must be positive");
			return -1.0;
		}
	
		double lk1 = 0;
		double lk2 = 0;
		//lk1 <- lgamma(0.5) + lgamma(nu/2) 
 		//- lgamma((nu+1)/2) + 0.5*log(nu)
 		//lk1 <- ifelse(nu < 1000000, lk1, 0.5*log(2*pi))
 		if (nu < 1000000) {
 			
			lk1 = Gamma.logGamma(0.5) 
					+ Gamma.logGamma(nu / 2) 
						- Gamma.logGamma((nu + 1) 
							/ 2) + 0.5 * FastMath.log(nu);
 		} else {
 			
 			lk1 = 0.5 * FastMath.log(2.0 * FastMath.PI);
 		}
 		
 		//lk2 <- ifelse(tau < 1000000, lk2, 0.5*log(2*pi))
		if (tau < 1000000) {
			
			//lk2 <- lgamma(0.5) + lgamma(tau/2) 
			//- lgamma((tau+1)/2) + 0.5*log(tau)
			lk2 = Gamma.logGamma(0.5) 
					+ Gamma.logGamma(tau / 2) 
					- Gamma.logGamma((tau + 1) / 2) 
								+ 0.5 * FastMath.log(tau);	
		} else {
			
			lk2 = 0.5 * FastMath.log(2.0 * FastMath.PI);
		}
 		
		//lk <- lk2 - lk1	
		//k <- exp(lk)
		final double k = FastMath.exp(lk2 - lk1);
		double out = 0;
		
		//cdf <- ifelse(q < mu, cdf1, cdf2)
		if (q < mu) {
			
			//cdf1 <- 2*pt((q-mu)/sigma, df=nu)
			tdDist.setDegreesOfFreedom(nu);
			out = 2.0 * tdDist.cumulativeProbability((q - mu) / sigma);
		} else {
			
			//cdf2 <- 1 + 2*k*(pt( (q-mu)/sigma, df=tau) - 0.5)
			tdDist.setDegreesOfFreedom(tau);
			out = 1 + 2.0 * k 
					* (tdDist.cumulativeProbability((q - mu) / sigma) - 0.5);
		}
		
		//cdf <- cdf/(1+k)
		out = out / (1 + k);
		
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
	 * pST4(q) launches pST4(q, mu, sigma, nu,  tau, lowerTail, isLog)
	 *  with deafult mu=0, sigma=1, nu=1, tau = 10.
	 * lowerTail = true, isLog = false.
	 * @param q - quantile
	 * @return value of cumulative probability function P(X <= q)
	 */
	//pST4 <- function(q, mu=0, sigma=1, nu=1,
	//tau=10, lower.tail = TRUE, log.p = FALSE)
	public final double pST4(final double  q) {
		return pST4(q, 0.0, 1.0, 1.0, 10.0, true, false);
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
	public final double qST4(final double p, 
	         				 final double mu, 
	         				 final double sigma, 
	         				 final double nu, 
	         				 final double tau, 
	         				 final boolean lowerTail, 
	         				 final boolean isLog) {
		
		// {  if (any(sigma < 0))stop(paste("sigma must be positive",))
		if (sigma < 0) {
			System.err.println("sigma must be positive");
			return -1.0;
		}
		
		//if (any(nu <= 0))  stop(paste("nu must be positive", "\n", ""))
		if (nu <= 0) {
			System.err.println("nu must be positive");
			return -1.0;
		}
		
		//if (any(tau < 0))  stop(paste("tau must be positive", "\n", ""))
		if (tau < 0) {
			System.err.println("tau must be positive");
			return -1.0;
		}
		
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
		
		double lk1 = 0;
		double lk2 = 0;
		//lk1 <- lgamma(0.5) + lgamma(nu/2) 
 		//- lgamma((nu+1)/2) + 0.5*log(nu)
 		//lk1 <- ifelse(nu < 1000000, lk1, 0.5*log(2*pi))
 		if (nu < 1000000) {
 			
			lk1 = Gamma.logGamma(0.5) 
					+ Gamma.logGamma(nu / 2.0) 
						- Gamma.logGamma((nu + 1) 
							/ 2.0) + 0.5 * FastMath.log(nu);
 		} else {
 			
 			lk1 = 0.5 * FastMath.log(2.0 * FastMath.PI);
 		}
 		
 		//lk2 <- ifelse(tau < 1000000, lk2, 0.5*log(2*pi))
		if (tau < 1000000) {
			
			//lk2 <- lgamma(0.5) + lgamma(tau/2) 
			//- lgamma((tau+1)/2) + 0.5*log(tau)
			lk2 = Gamma.logGamma(0.5) 
					+ Gamma.logGamma(tau / 2.0) 
					- Gamma.logGamma((tau + 1) / 2.0) 
								+ 0.5 * FastMath.log(tau);	
		} else {
			
			lk2 = 0.5 * FastMath.log(2.0 * FastMath.PI);
		}
 		
		//lk <- lk2 - lk1	
		//k <- exp(lk)
		final double k = FastMath.exp(lk2 - lk1);
		
		//q <- ifelse(p < (1/(1+k)), q1, q2)
		if (temp < (1/(1 + k))) {
				
			//q1 <- mu + sigma*qt(p*(1+k)/2, df=nu)
			tdDist.setDegreesOfFreedom(nu);
			out = mu + sigma * tdDist.inverseCumulativeProbability((
													temp * (1 + k)) / 2.0);
		} else {
				
			//q2 <- mu + sigma*qt((p*(1+k)-1)/(2*k) + 0.5, df=tau)
			tdDist.setDegreesOfFreedom(tau);
			out = mu + sigma * tdDist.inverseCumulativeProbability(((
									temp * (1 + k)) - 1) / (2.0 * k) + 0.5);
		}
		return out;
	}
	
	/**
	 * qST4(p) launches qST4(p, mu, sigma, nu,  tau, lowerTail, isLog)
	 *  with deafult mu=0, sigma=1, nu=1, tau = 10.
	 * lowerTail = true, isLog = false.
	 * @param p - value of cumulative probability 
	 * @return quantile
	 */
	//qST4 <- function(p, mu=0, sigma=1, nu=1,
	//tau=10, lower.tail = TRUE, log.p = FALSE)
	public final double qST4(final double  p) {
		return qST4(p,  0.0, 1.0, 1.0, 10.0, true, false);
	}	
	
	/** Generates a random sample from this distribution.
	 * @param mu -  vector of mu distribution parameters values
	 * @param sigma -  vector of sigma distribution parameters values
	 * @param nu -  vector of nu distribution parameters values
	 * @param tau -  vector of tau distribution parameters values
	 * @param uDist -  object of UniformRealDistribution class;
	 * @return random sample vector
	 */	
		public final double rST4(final double mu, 
		       	   				 final double sigma, 
		       	   				 final double nu, 
		       	   				 final double tau,
		       	   				 final UniformRealDistribution uDist) {
			
			// {  if (any(sigma <= 0))stop(paste("sigma must be positive",))
			if (sigma <= 0) {
				System.err.println("sigma must be positive");
				return -1.0;
			}
			
			//if (any(nu <= 0))  stop(paste("nu must be positive", "\n", ""))
			if (nu <= 0) {
				System.err.println("nu must be positive");
				return -1.0;
			}
			
			//if (any(tau <= 0))  stop(paste("tau must be positive", "\n", ""))
			if (tau <= 0) {
				System.err.println("tau must be positive");
				return -1.0;
			}
			    	    
			//r <- qST3(p,mu=mu,sigma=sigma,nu=nu,tau=tau)
			return qST4(uDist.sample(), mu, sigma, nu, tau, true, false);
		}
	
	/**
	* rST4(uDist) launches rST4(uDist, mu, sigma, nu,  tau)
 	* with deafult mu=0, sigma=1, nu=1, tau = 10.
	* @param uDist -  object of UniformRealDistribution class;
	* @return random sample value
	*/
	//rST4 <- function(n, mu=0, sigma=1, nu=1, tau=10)
	public final double rST4(final UniformRealDistribution uDist) {

		return rST4(0.0, 1.0, 1.0, 10.0, uDist);
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
		return nu.getMinValue() > 0;	
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
		return DistributionSettings.ST4;
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
