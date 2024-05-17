INFO = -Minfo=all
LIBS = -lboost_program_options
HOST = -acc=host
MULT = -acc=multicore
CXX = pgc++

all: onecore multicore
	

onecore: task.cpp
	$(CXX) $(HOST) $(INFO) $(LIBS) -o $@ $<

multicore: task.cpp
	$(CXX) $(MULT) $(INFO) $(LIBS) -o $@ $<

clean:all
	rm onecore multicore
