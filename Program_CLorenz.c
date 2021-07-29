#include <stdio.h>
#include <math.h>
#include <gmp.h>
#include <omp.h>
#define PREC 7488
#define N 2580
#define Numt 32
#define pad 8


int main()
{
    FILE *outfile;
    outfile = fopen("res_400.txt", "w");
    char ch[12];
    int i,j,l,tid,n,shift,sh;
    long int Exp;
    double d,t1,t2;

    mpf_set_default_prec(PREC);
    mpf_t Rs, Rf, a, c, ac, b, es, ef, tau, zero;
    mpf_t h1,h2,h2a,h3,h4,h5,h5a,h6;
    mpf_t u1,u2,time,time1,T,T1,temp,Abs;;
    mpf_init(h1);
    mpf_init(h2);
    mpf_init(h2a);
    mpf_init(h3);
    mpf_init(h4);
    mpf_init(h5);
    mpf_init(h5a);
    mpf_init(h6);
    mpf_init(T);
    mpf_init(T1);
    mpf_init(time1);
    mpf_init(temp);
    mpf_init(Abs);


    mpf_init_set_str(b,"0.0",10);
    mpf_init_set_str(zero,"0.0",10);
    mpf_init_set_str(time,"0.0",10);
    mpf_init_set_str(tau,"0.001",10);
    mpf_init_set_str(Rs,"28.0",10);
    mpf_init_set_str(Rf,"45.0",10);
    mpf_init_set_str(a,"10.0",10);
    mpf_init_set_str(c,"10.0",10);
    mpf_init_set_str(ac,"100.0",10);
    mpf_init_set_str(es,"0.01",10);
    mpf_init_set_str(ef,"10.0",10);
    mpf_init_set_str(u1,"8.0",10);
    mpf_init_set_str(u2,"3.0",10);
    mpf_div(u1,u1,u2);
    mpf_set(b,u1);
    mpf_clear(u1);
    mpf_clear(u2);

    mpf_t x[N+1],y[N+1],z[N+1],xf[N+1],yf[N+1],zf[N+1];

    for (i = 0; i<N+1; i++)
    {
       mpf_init(x[i]);
       mpf_init(y[i]);
       mpf_init(z[i]);
       mpf_init(xf[i]);
       mpf_init(yf[i]);
       mpf_init(zf[i]);


    }

    mpf_t sum[pad*Numt],tempv[pad*Numt];
    #pragma omp parallel for schedule(static)
    for (i = 0; i<pad*Numt; i++)
    {
        mpf_init_set(tempv[i],zero);
    }

    #pragma omp parallel for schedule(static)
    for (i = 0; i<pad*Numt; i++)
    {
        mpf_init_set(sum[i],zero);
    }

    mpf_set_str(T1,"10.0",10);
    mpf_set_str(time1,"10.0",10);
    mpf_set_str(T,"400.001",10);
    mpf_set_str(x[0],"5.0",10);
    mpf_set_str(y[0],"5.0",10);
    mpf_set_str(z[0],"10.0",10);
    mpf_set_str(xf[0],"5.0",10);
    mpf_set_str(yf[0],"5.0",10);
    mpf_set_str(zf[0],"10.0",10);

    mpf_set_str(time,"0.0",10);
    double start = omp_get_wtime();


    l=0;
    while (mpf_cmp(time,T)<0)
    {
    #pragma omp parallel private(i,j,tid,n,shift,sh)
    {
        tid = omp_get_thread_num();
        for (i = 0; i<N; i++)
        {

            # pragma omp for schedule(static)
            for (j=0; j<=i; j++)
            {
                mpf_mul(tempv[pad*tid],x[i-j],z[j]);
                mpf_add(sum[pad*tid],sum[pad*tid],tempv[pad*tid]);

                mpf_mul(tempv[pad*tid],x[i-j],y[j]);
                mpf_add(sum[pad*tid+1],sum[pad*tid+1],tempv[pad*tid]);

                mpf_mul(tempv[pad*tid],xf[i-j],yf[j]);
                mpf_add(sum[pad*tid+2],sum[pad*tid+2],tempv[pad*tid]);

                mpf_mul(tempv[pad*tid],xf[i-j],zf[j]);
                mpf_add(sum[pad*tid+3],sum[pad*tid+3],tempv[pad*tid]);

                mpf_mul(tempv[pad*tid],xf[i-j],y[j]);
                mpf_add(sum[pad*tid+4],sum[pad*tid+4],tempv[pad*tid]);

            }
            //! Explicit Parallel Reduction for two sums for log(p) additions
            //! The first step is in a butterfly form
            n=Numt;
            shift=(n+1)/2;
            if (tid <=n-1-shift)
            {
                 mpf_add(sum[pad*tid],sum[pad*tid],sum[pad*(tid+shift)]);
                 mpf_add(sum[pad*tid+1],sum[pad*tid+1],sum[pad*(tid+shift)+1]);
                 mpf_add(sum[pad*tid+2],sum[pad*tid+2],sum[pad*(tid+shift)+2]);
            }
            else if (tid>=shift && tid < n)
            {
                 mpf_add(sum[pad*tid+3],sum[pad*tid+3],sum[pad*(tid-shift)+3]);
                 mpf_add(sum[pad*tid+4],sum[pad*tid+4],sum[pad*(tid-shift)+4]);
            }
            sh=n-shift;
            n=shift;
            shift=(n+1)/2;
            # pragma omp barrier
            while (n>1)
            {
                  if (tid <=n-1-shift)
                  {
                        mpf_add(sum[pad*tid],sum[pad*tid],sum[pad*(tid+shift)]);
                        mpf_add(sum[pad*tid+1],sum[pad*tid+1],sum[pad*(tid+shift)+1]);
                        mpf_add(sum[pad*tid+2],sum[pad*tid+2],sum[pad*(tid+shift)+2]);

                  }
                  else if (tid>=sh && tid<=sh+n-1-shift)
                  {
                        mpf_add(sum[pad*tid+3],sum[pad*tid+3],sum[pad*(tid+shift)+3]);
                        mpf_add(sum[pad*tid+4],sum[pad*tid+4],sum[pad*(tid+shift)+4]);
                  }
                  n=shift;
                  shift=(n+1)/2;
                  # pragma omp barrier
            }
            /// End of explicit Parallel Reduction for two sums for log(p) additions
             #pragma omp sections
             {
             #pragma omp section
             {
             mpf_sub(h1,y[i],x[i]);
             mpf_mul(h1,h1,a);
             mpf_div_ui(x[i+1],h1,i+1);
             }

             #pragma omp section
             {
             mpf_mul(h2,Rs,x[i]);
             mpf_sub(h2,h2,y[i]);
             mpf_sub(h2,h2,sum[0]);
             mpf_mul(h2a,es,sum[2]);
             mpf_sub(h2,h2,h2a);
             mpf_div_ui(y[i+1],h2,i+1);
             }

             #pragma omp section
             {
             mpf_mul(h3,b,z[i]);
             mpf_sub(h3,sum[1],h3);
             mpf_div_ui(z[i+1],h3,i+1);
             }

             #pragma omp section
             {
               mpf_sub(h4,yf[i],xf[i]);
               mpf_mul(h4,h4,ac);
               mpf_div_ui(xf[i+1],h4,i+1);
             }

             #pragma omp section
             {
                mpf_mul(h5,Rf,xf[i]);
                mpf_sub(h5,h5,yf[i]);
                mpf_sub(h5,h5,sum[pad*sh+3]);
                mpf_mul(h5,h5,c);
                mpf_mul(h5a,ef,sum[pad*sh+4]);
                mpf_add(h5,h5,h5a);
                mpf_div_ui(yf[i+1],h5,i+1);
             }

             #pragma omp section
             {
                mpf_mul(h6,b,zf[i]);
                mpf_sub(h6,sum[2],h6);
                mpf_mul(h6,h6,c);
                mpf_div_ui(zf[i+1],h6,i+1);
             }
             }

             mpf_set(sum[pad*tid],zero);
             mpf_set(sum[pad*tid+1],zero);
             mpf_set(sum[pad*tid+2],zero);
             mpf_set(sum[pad*tid+3],zero);
             mpf_set(sum[pad*tid+4],zero);
        }
        //! determining the step size and weather to print;
        #pragma omp single
        {
           //-------------------------------------------
           mpf_abs (Abs, x[N-1]);
           mpf_set(temp,Abs);

           mpf_abs (Abs, y[N-1]);
           if(mpf_cmp (Abs, temp)>0) mpf_set(temp,Abs);

           mpf_abs (Abs, z[N-1]);
           if(mpf_cmp (Abs, temp)>0) mpf_set(temp,Abs);

           mpf_abs (Abs, xf[N-1]);
           if(mpf_cmp (Abs, temp)>0) mpf_set(temp,Abs);

           mpf_abs (Abs, yf[N-1]);
           if(mpf_cmp (Abs, temp)>0) mpf_set(temp,Abs);

           mpf_abs (Abs, zf[N-1]);
           if(mpf_cmp (Abs, temp)>0) mpf_set(temp,Abs);


           mpf_get_str (ch, &Exp, 2, 10, temp);
           mpf_div_2exp (temp, temp, Exp);
           d=1.0/mpf_get_d (temp);
           t1=pow(d,1.0/((double)(N-1)));
           t1=t1*pow(2.0,-Exp/(double)(N-1));
           //-------------------------------------------
           mpf_abs (Abs, x[N]);
           mpf_set(temp,Abs);

           mpf_abs (Abs, y[N]);
           if(mpf_cmp (Abs, temp)>0) mpf_set(temp,Abs);

           mpf_abs (Abs, z[N]);
           if(mpf_cmp (Abs, temp)>0) mpf_set(temp,Abs);

           mpf_abs (Abs, xf[N]);
           if(mpf_cmp (Abs, temp)>0) mpf_set(temp,Abs);

           mpf_abs (Abs, yf[N]);
           if(mpf_cmp (Abs, temp)>0) mpf_set(temp,Abs);

           mpf_abs (Abs, zf[N]);
           if(mpf_cmp (Abs, temp)>0) mpf_set(temp,Abs);

           mpf_get_str (ch, &Exp, 2, 10, temp);
           mpf_div_2exp (temp, temp, Exp);
           d=1.0/mpf_get_d (temp);
           t2=pow(d,1.0/((double)(N)));
           t2=t2*pow(2.0,-Exp/(double)(N));
           //-------------------------------------------
           if(t2<t1) t1=t2;
           t1=t1/exp(2.0);
           t1=t1*0.993;
           mpf_set_d (tau, t1);
           mpf_add(temp,time,tau);
           if(mpf_cmp (temp, time1)>=0)
           {
               mpf_sub(tau, time1, time);
               mpf_add(time1, time1, T1);
               l=1;
           }

         }
        //! end of determining the step size and weather to print;
        //! One step forward with Horner's rule
        #pragma omp sections
        {
              #pragma omp section
              {
                     mpf_set(h1,x[N]);
                     for (j=N-1; j>=0; j--)
                     {
                         mpf_mul(h1,h1,tau);
                         mpf_add(h1,h1,x[j]);
                     }
                     mpf_set(x[0],h1);
              }
              #pragma omp section
              {
                     mpf_set(h2,y[N]);
                     for (j=N-1; j>=0; j--)
                     {
                         mpf_mul(h2,h2,tau);
                         mpf_add(h2,h2,y[j]);
                     }
                     mpf_set(y[0],h2);
              }
              #pragma omp section
              {
                     mpf_set(h3,z[N]);
                     for (j=N-1; j>=0; j--)
                     {
                         mpf_mul(h3,h3,tau);
                         mpf_add(h3,h3,z[j]);
                     }
                     mpf_set(z[0],h3);
              }

              #pragma omp section
              {
                     mpf_set(h4,xf[N]);
                     for (j=N-1; j>=0; j--)
                     {
                         mpf_mul(h4,h4,tau);
                         mpf_add(h4,h4,xf[j]);
                     }
                     mpf_set(xf[0],h4);
              }
              #pragma omp section
              {
                     mpf_set(h5,yf[N]);
                     for (j=N-1; j>=0; j--)
                     {
                         mpf_mul(h5,h5,tau);
                         mpf_add(h5,h5,yf[j]);
                     }
                     mpf_set(yf[0],h5);
              }
              #pragma omp section
              {
                     mpf_set(h6,zf[N]);
                     for (j=N-1; j>=0; j--)
                     {
                         mpf_mul(h6,h6,tau);
                         mpf_add(h6,h6,zf[j]);
                     }
                     mpf_set(zf[0],h6);
              }

        }
    }

        mpf_add(time,time,tau);
        if(l==1)
        {
             mpf_out_str(outfile,10,10,time);
             fprintf(outfile,"\n");
             mpf_out_str(outfile,10,60,x[0]);
             fprintf(outfile,"\n");
             mpf_out_str(outfile,10,60,y[0]);
             fprintf(outfile,"\n");
             mpf_out_str(outfile,10,60,z[0]);
             fprintf(outfile,"\n");
             mpf_out_str(outfile,10,60,xf[0]);
             fprintf(outfile,"\n");
             mpf_out_str(outfile,10,60,yf[0]);
             fprintf(outfile,"\n");
             mpf_out_str(outfile,10,60,zf[0]);
             fprintf(outfile,"\n");
             l=0;
        }

    }


    printf("Time = %f.\n",omp_get_wtime()-start);


    mpf_clear(h1);
    mpf_clear(h2);
    mpf_clear(h2a);
    mpf_clear(h3);
    mpf_clear(h4);
    mpf_clear(h5);
    mpf_clear(h5a);
    mpf_clear(h6);
    mpf_clear(T);
    mpf_clear(zero);
    mpf_clear(time);
    mpf_clear(tau);
    mpf_clear(Rs);
    mpf_clear(Rf);
    mpf_clear(a);
    mpf_clear(c);
    mpf_clear(ac);
    mpf_clear(b);
    mpf_clear(es);
    mpf_clear(ef);
    mpf_clear(time1);
    mpf_clear(T1);
    mpf_clear(temp);
    mpf_clear(Abs);


    for (i = 0; i<N+1; i++)
    {
        mpf_clear(x[i]);
        mpf_clear(y[i]);
        mpf_clear(z[i]);
        mpf_clear(xf[i]);
        mpf_clear(yf[i]);
        mpf_clear(zf[i]);
    }

    for (i = 0; i<pad*Numt; i++)
    {
        mpf_clear(tempv[i]);

    }

    for (i = 0; i<pad*Numt; i++)
    {
        mpf_clear(sum[i]);

    }

    fclose(outfile);


    return 0;
}






