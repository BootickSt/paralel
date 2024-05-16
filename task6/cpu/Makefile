INFO = -Minfo=all
LIBS = -lboost_program_options
HOST = -acc=host
MULT = -acc=multicore
CXX = pgc++

all:serial multi
	

serial:
	$(CXX) $(HOST) $(INFO) $(LIBS) -o $@ task.cpp

multi:
	$(CXX) $(MULT) $(INFO) $(LIBS) -o $@ task.cpp

clean:all
	rm serial multi
