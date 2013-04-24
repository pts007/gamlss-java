/*
  Copyright 2012 by Dr. Vlasios Voudouris and ABM Analytics Ltd
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/
package gamlss.utilities.oi;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.StringTokenizer;

import javax.swing.JOptionPane;

public class CSVFileReader 
{
	private  String fileName="data/oilAll.csv";	
	public  ArrayList <String>storeValues = new ArrayList<String>();
	
	 public CSVFileReader(String fileName)
	{
		this.fileName=fileName;
	}
	
	public CSVFileReader(){}
	
	public void readFile()
	{
		try {	
			  storeValues.clear();//just in case this is the second call of the ReadFile Method./
			  BufferedReader br = new BufferedReader( new FileReader(fileName));
			  String line;
		   while( (line = br.readLine()) != null)
		   {
				storeValues.add(line);	
		   }
		   br.close();
		   br=null;
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	//mutators and accesors 
	public void setFileName(String newFileName)
	{
		this.fileName=newFileName;
	}
	
	public String getFileName()
	{
		return fileName;
	}
	
	public ArrayList getFileValues()
	{
		return this.storeValues;
	}
	
	public void displayArrayList()
	{
		for(int x=0;x<this.storeValues.size();x++)
		{
			//Make sure ALL rows have the same columns which specified by the first row! 
			if (storeValues.get(x).split(",").length != storeValues.get(0).split(",").length)
			{
				JOptionPane.showMessageDialog(null,"error with number of colums");
				System.out.println((Arrays.toString(storeValues.get(x).split(","))));
				return;
			}
			else
			{
				//System.out.println(Arrays.toString(storeValues.get(x).split(",")));
				String[] line = storeValues.get(x).split(",");
				
				for (int i=0; i<line.length;i++)
				{
					System.out.println("Index:"+i + " text:"+ line[i]);
				}
			}
		}
	}
	
	public static void main(String[] args) 
	{
			CSVFileReader x=new CSVFileReader("data/Countries.csv");
			x.readFile();
			x.displayArrayList();	
	}


	
}