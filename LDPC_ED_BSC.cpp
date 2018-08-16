//
//  LDPC_ED_BSC.cpp
//  
//
//  Created by iLab on 5/2/16.
//
//


#include <sstream>
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <time.h>
#include <cstdlib>
#include <cmath>
#include <string>
/*#define N 512
#define K 466
#define M 48*/

#define N 4376
#define K 4095
#define M 282

using namespace std;

    int G[K][N];
    int H[M][N];
    int message[K];
    int codeword[N];
    int bpskword[N];
    int rbpskword[N];
    int decodedcw[N];
    int mc_map[K];
    double F[N]; //as in paper
    double F_deep[N];
    double z[M][N];
    int zlin[N];
    double T[M][N];
    double L[M][N];
    int k;
    float pu;
    void sendchannel();
    void decode();
    float alpha;


void find_FDeep(int l)
{
    string filename = "../results_f/html/0p01/"+to_string(l)+".csv"; //change here
    ifstream myfile (filename);
    string value;
    int count = 0;
    for(int i = 0; i < N; i++)
    {
        F_deep[i] = 0.00;
    }
    while ( myfile.good() )
    {
        getline ( myfile, value, '\n' ); // read a string until next comma: http://www.cplusplus.com/reference/string/getline/
        
            //cout<<value<<endl;
        if(count != 4095)
        {
            double val = stod(value);
            if (val < 1e-10)
            {
                val = 1e-10;
            }
            else if(val > 1.0-1e-10)
            {
                val = 1.0-1e-10;
            }
            F_deep[mc_map[count]] = -log((1.0-val)/val);
            
            //cout<<val<<" :  "<<log((1.0-val)/val)<<endl;
        }
        
        count++;
        
    }
    //exit(0);
}

void decode(int l)
{
    find_FDeep(l);
    for(int j = 0; j <N; j++)
    {
        F[j] = 0;
        zlin[j] = 0;
        
        for(int i = 0; i <M; i++)
        {
            z[i][j] = 0;
            L[i][j] = 0;
            T[i][j] = 0;
        }
    }
    //Initialize
    alpha = 1; //change here
    for(int j = 0;j <N; j++)
    {
        F[j] = (-pow(-1,(rbpskword[j]+1)/2)*log((1-pu)/pu))+(alpha*F_deep[j]);
        //cout<<alpha*F_deep[j]<<endl;
    }
    
  
    
    
    for(int i = 0; i <M; i++)
    {
        
        for(int j = 0; j <N; j++)
        {
            if(H[i][j] == 1)
                z[i][j] = F[j]; //variable node j to check node i
        }
    }
    
   
    int t = 0;
    bool undecoded = true;
    
    while(t < 10 && undecoded)
    {
        
        
        
        for(int j = 0; j <N; j++)
        {
            
            for(int i = 0; i <M; i++)
            {
                if(H[i][j] == 1)
                {
                    T[i][j] = 1;
                    for(int j1 = 0; j1 <N; j1++)
                    {
                        if(j1 != j && H[i][j1]  == 1)
                        {
                            double ele;
                            ele = (1.0-exp(z[i][j1]))/(1.0+exp(z[i][j1]));
                            T[i][j] = T[i][j]*ele;
                        }
                    }
                    double ele2;
                    ele2 = (1-T[i][j])/(1+T[i][j]);
                    L[i][j] = log(ele2);
                }
                
            }
            
        }
        //checknode
      
        
        
        //variable node
        for(int j = 0; j <N; j++)
        {
            for(int i = 0; i <M; i++)
            {
                if(H[i][j] == 1)
                {
                    z[i][j] = F[j];
                    for(int i1 = 0; i1 <M; i1++)
                    {
                        if(H[i1][j] == 1 && i1 != i)
                            z[i][j] = z[i][j]+L[i1][j];
                    }
                }
            }
            
        }
        
        
        for(int j = 0; j <N; j++)
        {
            zlin[j] = F[j];
            for(int i = 0; i <M; i++)
            {
                if(H[i][j] == 1)
                {
                    zlin[j] = zlin[j]+L[i][j];
                    //NSLog(@"%f",zlin[j]);
                }
            }
        }
        
        
        
        
        
        
        //stopping criteria
        for(int j =0; j<N; j++)
        {
            if(zlin[j] < 0)
            {
                decodedcw[j] = 0;
            }
            else
            {
                decodedcw[j] = 1;
            }
            //NSLog(@"%d",decodedcw[j]);
        }
        
        int zero[M] = {0};
        
        for(int i = 0; i < M; i++)
        {
            zero[i] = 0;
            for(int j = 0; j < N; j++)
            {
                zero[i] = (zero[i]+decodedcw[j]*H[i][j])%2;
            }
            
        }
        
        int count = 0;
        for(int i = 0; i < M; i++)
        {
            if(zero[i] != 0)
                count++;
        }
        
        if(count == 0)
            undecoded = false;
        
        t++;
        
    }
    
}



void sendchannel(int l)
{
    int cou = 0;
    for(int i = 0; i < N; i++)
    {
        float randomN = rand()*1.0/RAND_MAX;
        //NSLog(@"random %f",randomN*10);
        if(randomN <= pu)
        {
            cou++;
            if(bpskword[i] == -1)
            {
                rbpskword[i] = 1;
                
            }
            if(bpskword[i] == 1)
            {
                rbpskword[i] = -1;
            }
            
            
            
        }
        else
        {
            rbpskword[i] = bpskword[i];
        }
        
        
    }
    
    //Overwrite with preexisting data
    
    string line;
    string filename = "../4095_noisy_randomized_0p01/testing/html/"+to_string(l)+".txt"; //change here
   // cout<<filename<<endl;
    ifstream myfile (filename);
    if (myfile.is_open())
    {
        while ( getline (myfile,line) )
        {
            
            for (int i=0; i<line.length(); ++i)
            {
                int mess = line[i]- '0';
                int bps_mess = 2*mess-1;
               // cout<<i<<" "<<message[i]<<" "<<codeword[mc_map[i]]<<" "<<bpskword[mc_map[i]] <<" : "<<bps_mess<<endl;
                rbpskword[mc_map[i]] = bps_mess;
                
            }
            
        }
        myfile.close();
    }
    
    else cout << "Unable to open file: "+filename<<endl;
    
    
}

void readmessage(int l)
{
    
    
    string line;
    string filename = "../4095_randomized/testing/html/"+to_string(l)+".txt"; //change
    ifstream myfile (filename);
    if (myfile.is_open())
    {
        while ( getline (myfile,line) )
        {
            
            for (int i=0; i<line.length(); ++i)
            {
                message[i] = line[i]- '0';
                
            }
            
        }
        myfile.close();
    }
    
    else cout << "Unable to open file: "+filename<<endl;
    
    
}

void prepare_mcmap()
{
    string filename = "msg_cw_map.txt";
    ifstream myfile (filename);
    string value;
    int count = 0;
    while ( myfile.good() )
    {
        getline ( myfile, value, ',' ); // read a string until next comma: http://www.cplusplus.com/reference/string/getline/
        if(count != 8189)
        {
            //cout<<value<<endl;
            int val = stoi(value);
            if (count != 0)
            {
                mc_map[count-1] = val;
                //cout<<count-1<<" : "<<val<<endl;
            }
        }
        count++;
        
    }
    //exit(0);
}
int main()
{
    
    
    ifstream file("generatorM.txt");
    string line;
    int counti = 0;
    int countj = 0;
    while(getline(file,line))
    {
        istringstream iss(line);
        int value;
        while(iss >> value)
        {
            G[counti][countj] = value;
            countj++;
            
        }
        //cout<<"j->"<<countj<<endl;
        countj = 0;
        counti++;
        
        //cout<<"i->"<<counti<<endl;
    }
    
    ifstream file1("parityM.txt");
    string line1;
    int counti1 = 0;
    int countj1 = 0;
    while(getline(file1,line1))
    {
        istringstream iss1(line1);
        int value1;
        while(iss1 >> value1)
        {
            H[counti1][countj1] = value1;
            countj1++;
            
        }
        //cout<<"j->"<<countj1<<endl;
        countj1 = 0;
        counti1++;
        
        // cout<<"i->"<<counti1<<endl;
    }
    
    //cout<<endl;
    
        /*for(int i =0; i < N; i++ )
     {
     cout<<codeword[i];
     }
     cout<<endl;*/
    int count;
    int errcount;
    int tcount, terrcount;
    int framecount;
    pu  = 0.01; //change here
    framecount = 0;
    tcount = 0;
    terrcount = 0;
    int numcodes = 1000;
    prepare_mcmap();
    for(int l = 0; l<numcodes; l++)
    {
        
        //cout<<l<<endl;
        srand (time(NULL));
        
            
        readmessage(l);
        
        for(int i =0; i < N; i++ )
        {
            codeword[i] = 0;
            for(int j =0; j < K; j++ )
            {
                codeword[i] =(codeword[i]+ message[j]*G[j][i])%2;
            }
            
        }
        
        for(int i =0; i < N; i++ )
        {
            bpskword[i] =  2*codeword[i]-1;
        }
        errcount = 0;
        sendchannel(l);
        //cout<<errcount<<endl;
        terrcount = terrcount+errcount;
       
       
        //if(l%10 == 0)
           // cout<<"Codeword set"<<l<<endl;
        decode(l);
        count = 0;
        for(int i =0; i < N; i++ )
        {
            if(decodedcw[i] != codeword[i])
                count++;
        }
        
        tcount = count+tcount;
        
       
        
        if(count != 0)
            framecount++;
        
        cout<<"p: "<<pu<<", alpha: "<<alpha<<", frames  "<<l+1<<"  , success percentage "<<100.0-(framecount*100.0/(l+1))<<endl;
        
    }
    //cout<<tcount<<endl;
    cout<<"Frame decoding success percentage :   "<<100.0-framecount*100.0/numcodes<<endl;
    //cout<<100.0-tcount*100.0/terrcount<<endl;
    return 0;
    
    
}
