F:
call F:\opt_py\env1\Scripts\activate.bat
cd F:\opt_py


python opt_run3.py -p 5 -b 32 -e 50 -d 10 +i -v 0.94 -G

python opt_run3.py -p 5 -b 64 -e 50 -d 10 +i -v 0.94

python opt_run3.py -p 5 -b 128 -e 50 -d 10 +i -v 0.94


python opt_run3.py -p 5 -b 32 -e 50 -d 100 +i -v 0.7495 -G

python opt_run3.py -p 5 -b 64 -e 50 -d 100 +i -v 0.7555

python opt_run3.py -p 5 -b 128 -e 50 -d 100 +i