#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

int transform_pos(int w, int h){
	return ((h + 5) * 25) + (w + 5);
}

int transform_pos_norm(int w, int h){
	return ((h) * 15) + (w);
}


void get_points(double* data, int *to_look, int counter, int cur_player, double *points) {
/*
	char pattern_first[20][10] = {"11111", "011110", "01111", "11110","010111","011011","011101",
	"111010","110110","101110","01110","0111","1110",
	"01101","01011","11010","10110","0110","10","01"};

	char pattern_second[20][10] = {"22222", "022220", "02222", "22220","020222","022022","022202",
	"222020","220220","202220","02220","0222","2220",
	"02202","02022","22020","20220","0220","20","02"};

	long long cost[20] = {999999999, 7000, 4000, 4000, 2500, 2500, 2500,
	 2500, 2500, 2500, 3000, 1500, 1500,
	  2000, 2000, 2000, 2000, 200, 50, 50};

*/

	char pattern_first[20][10] = {"11111", "011110", "01111", "11110","010111","011011","011101",
	"111010","110110","101110","01110","0111","1110",
	"01101","01011","11010","10110","0110","10","01"};

	char pattern_second[20][10] = {"22222", "022220", "02222", "22220","020222","022022","022202",
	"222020","220220","202220","02220","0222","2220",
	"02202","02022","22020","20220","0220","20","02"};

	long long cost[20] = {999999999, 7000, 4000, 4000, 2500, 2500, 2500,
	 2500, 2500, 2500, 3000, 1500, 1500,
	  2000, 2000, 2000, 2000, 200, 50, 50};


	for(int elem = 0; elem < counter; elem++){

        if (to_look[elem] < 0 || to_look[elem] > 224){
            continue;
        }
		int w = to_look[elem] % 15;
		int h = to_look[elem] / 15;

		int line[11] = {
							data[transform_pos(w + 5, h + 5)],
							data[transform_pos(w + 4, h + 4)],
							data[transform_pos(w + 3, h + 3)],
							data[transform_pos(w + 2, h + 2)],
							data[transform_pos(w + 1, h + 1)],
							data[transform_pos(w, h)],
							data[transform_pos(w - 1, h - 1)],
							data[transform_pos(w - 2, h - 2)],
							data[transform_pos(w - 3, h - 3)],
							data[transform_pos(w - 4, h - 4)],
							data[transform_pos(w - 5, h - 5)]
						};
		int positions[11] = {
			transform_pos_norm(w + 5, h + 5),
			transform_pos_norm(w + 4, h + 4),
			transform_pos_norm(w + 3, h + 3),
			transform_pos_norm(w + 2, h + 2),
			transform_pos_norm(w + 1, h + 1),
			transform_pos_norm(w, h),
			transform_pos_norm(w - 1, h - 1),
			transform_pos_norm(w - 2, h - 2),
			transform_pos_norm(w - 3, h - 3),
			transform_pos_norm(w - 4, h - 4),
			transform_pos_norm(w - 5, h - 5),
		};

		void add_points() {

			for(int pos_check = 0; pos_check < 20; ++pos_check){

				for (int i = 0; i + strlen(pattern_first[pos_check]) < 11; ++i){
					int flag = 1;
					for (int j = i; j < i + strlen(pattern_first[pos_check]); ++j)
					{
						if (line[j] != (pattern_first[pos_check][j - i] - '0')) {
							flag = 0;
							break;
						}
					}
					if (flag) {
						for (int k = i; k < i + strlen(pattern_first[pos_check]); ++k){
							if (positions[k] > 0 && positions[k] < 225){
								if (cur_player == 1){
									points[positions[k]] += cost[pos_check] + (cost[pos_check] / 10);
								} else {
									points[positions[k]] += cost[pos_check];
								}
							}
						}
					}
				}

				for (int i = 0; i + strlen(pattern_second[pos_check]) < 11; ++i){
					int flag = 1;
					for (int j = i; j < i + strlen(pattern_second[pos_check]); ++j)
					{
						if (line[j] != (pattern_second[pos_check][j - i] - '0')) {
							flag = 0;
							break;
						}
					}
					if (flag) {
						for (int k = i; k < i + strlen(pattern_second[pos_check]); ++k){
							if (positions[k] > 0 && positions[k] < 225){
								if (cur_player == 2){
									points[positions[k]] += cost[pos_check] + (cost[pos_check] / 10);
								} else {
									points[positions[k]] += cost[pos_check];
								}
							}
						}
					}
				}

			}


		}


		add_points();

//-----------------------------------------------------------------------------------------------------------------------------

		

		line[0] = data[transform_pos(w, h + 5)];
		line[1] = data[transform_pos(w, h + 4)];
		line[2] = data[transform_pos(w, h + 3)];
		line[3] = data[transform_pos(w, h + 2)];
		line[4] = data[transform_pos(w, h + 1)];
		line[5] = data[transform_pos(w, h)];
		line[6] = data[transform_pos(w, h - 1)];
		line[7] = data[transform_pos(w, h - 2)],
		line[8] = data[transform_pos(w, h - 3)];
		line[9] = data[transform_pos(w, h - 4)];
		line[10] = data[transform_pos(w, h - 5)];
						
		positions[0] = transform_pos_norm(w, h + 5);
		positions[1] = transform_pos_norm(w, h + 4);
		positions[2] = transform_pos_norm(w, h + 3);
		positions[3] = transform_pos_norm(w, h + 2);
		positions[4] = transform_pos_norm(w, h + 1);
		positions[5] = transform_pos_norm(w, h);
		positions[6] = transform_pos_norm(w, h - 1);
		positions[7] = transform_pos_norm(w, h - 2);
		positions[8] = transform_pos_norm(w, h - 3);
		positions[9] = transform_pos_norm(w, h - 4);
		positions[10] = transform_pos_norm(w, h - 5);


		add_points();


//-----------------------------------------------------------------------------------------------------------------------

		
		line[0] = data[transform_pos(w - 5, h + 5)];
		line[1] = data[transform_pos(w - 4, h + 4)];
		line[2] = data[transform_pos(w - 3, h + 3)];
		line[3] = data[transform_pos(w - 2, h + 2)];
		line[4] = data[transform_pos(w - 1, h + 1)];
		line[5] = data[transform_pos(w, h)];
		line[6] = data[transform_pos(w + 1, h - 1)];
		line[7] = data[transform_pos(w + 2, h - 2)];
		line[8] = data[transform_pos(w + 3, h - 3)];
		line[9] = data[transform_pos(w + 4, h - 4)];
		line[10] = data[transform_pos(w + 5, h - 5)];
						

		positions[0] = 	transform_pos_norm(w - 5, h + 5);
		positions[1] = 	transform_pos_norm(w - 4, h + 4);
		positions[2] = 	transform_pos_norm(w - 3, h + 3);
		positions[3] = 	transform_pos_norm(w - 2, h + 2);
		positions[4] = 	transform_pos_norm(w - 1, h + 1);
		positions[5] = 	transform_pos_norm(w, h);
		positions[6] = 	transform_pos_norm(w + 1, h - 1);
		positions[7] = 	transform_pos_norm(w + 2, h - 2);
		positions[8] = 	transform_pos_norm(w + 3, h - 3);
		positions[9] = 	transform_pos_norm(w + 4, h - 4);
		positions[10] = transform_pos_norm(w + 5, h - 5);


		add_points();


//--------------------------------------------------------------------------------------------------------------------------
		
						
		line[0] = data[transform_pos(w - 5, h)];
		line[1] = data[transform_pos(w - 4, h)];
		line[2] = data[transform_pos(w - 3, h)];
		line[3] = data[transform_pos(w - 2, h)];
		line[4] = data[transform_pos(w - 1, h)];
		line[5] = data[transform_pos(w, h)];
		line[6] = data[transform_pos(w + 1, h)];
		line[7] = data[transform_pos(w + 2, h)];
		line[8] = data[transform_pos(w + 3, h)];
		line[9] = data[transform_pos(w + 4, h)];
		line[10] = data[transform_pos(w + 5, h)];
						

		positions[0]	= transform_pos_norm(w - 5, h);
		positions[1]	= transform_pos_norm(w - 4, h);
		positions[2]	= transform_pos_norm(w - 3, h);
		positions[3]	= transform_pos_norm(w - 2, h);
		positions[4]	= transform_pos_norm(w - 1, h);
		positions[5]	= transform_pos_norm(w, h);
		positions[6]	= transform_pos_norm(w + 1, h);
		positions[7]	= transform_pos_norm(w + 2, h);
		positions[8]	= transform_pos_norm(w + 3, h);
		positions[9]	= transform_pos_norm(w + 4, h);
		positions[10]	= transform_pos_norm(w + 5, h);

		add_points();
//----------------------------------------------------------------------------------------------------------------------------
	}
}

void get_policy(double *data, int cur_player, double *points) {
	/*
	FILE* test;
	test = fopen("testik.txt", "w+");

	for (int i = 0; i < 15; ++i)
	{
		for (int j = 0; j < 15; ++j)
		{
			fprintf(test, "%.1lf ", data[transform_pos(j, i)]);
		}
		fprintf(test, "\n");
	}
	fclose(test);
	*/

	int to_look_mine[6250], to_look_opp[6250];

	int look_mine_counter = 0, look_opp_counter = 0;


	for (int h = 0;  h < 15; ++h){
		for (int w = 0; w < 15; ++w)
		{
			/*
			if (data[transform_pos(w, h)] != 0){
				if (data[transform_pos(w, h)] == cur_player){
					to_look_mine[look_mine_counter] = (h * 15) + w;
					look_mine_counter++;
			
					if (w > 1 && h > 1 && w < 13 && h < 13){
						to_look_mine[look_mine_counter] = ((h + 1) * 15) + (w + 1);
						look_mine_counter++;

						to_look_mine[look_mine_counter] = ((h + 1) * 15) + (w);
						look_mine_counter++;

						to_look_mine[look_mine_counter] = ((h + 1) * 15) + (w - 1);
						look_mine_counter++;

						to_look_mine[look_mine_counter] = ((h ) * 15) + (w - 1);
						look_mine_counter++;

						to_look_mine[look_mine_counter] = ((h - 1) * 15) + (w - 1);
						look_mine_counter++;

						to_look_mine[look_mine_counter] = ((h - 1) * 15) + (w);
						look_mine_counter++;

						to_look_mine[look_mine_counter] = ((h - 1) * 15) + (w + 1);
						look_mine_counter++;




                        if (w > 3 && h > 3 && w < 11 && h < 11){

						    to_look_mine[look_mine_counter] = ((h + 2) * 15) + (w + 2);
						    look_mine_counter++;

						    to_look_mine[look_mine_counter] = ((h + 2) * 15) + (w);
						    look_mine_counter++;

						    to_look_mine[look_mine_counter] = ((h + 2) * 15) + (w - 2);
						    look_mine_counter++;

						    to_look_mine[look_mine_counter] = ((h ) * 15) + (w - 2);
						    look_mine_counter++;

						    to_look_mine[look_mine_counter] = ((h - 2) * 15) + (w - 2);
						    look_mine_counter++;

						    to_look_mine[look_mine_counter] = ((h - 2) * 15) + (w);
						    look_mine_counter++;

						    to_look_mine[look_mine_counter] = ((h - 2) * 15) + (w + 2);
						    look_mine_counter++;
                        }
					}
					
				} else {
					to_look_opp[look_opp_counter] = (h * 15) + w;
					look_opp_counter++;

					if (w > 1 && h > 1 && w < 13 && h < 13){
						to_look_opp[look_opp_counter] = ((h + 1) * 15) + (w + 1);
						look_opp_counter++;

						to_look_opp[look_opp_counter] = ((h + 1) * 15) + (w);
						look_opp_counter++;

						to_look_opp[look_opp_counter] = ((h + 1) * 15) + (w - 1);
						look_opp_counter++;

						to_look_opp[look_opp_counter] = ((h ) * 15) + (w - 1);
						look_opp_counter++;

						to_look_opp[look_opp_counter] = ((h - 1) * 15) + (w - 1);
						look_opp_counter++;

						to_look_opp[look_opp_counter] = ((h - 1) * 15) + (w);
						look_opp_counter++;

						to_look_opp[look_opp_counter] = ((h - 1) * 15) + (w + 1);
						look_opp_counter++;


                        if (w > 3 && h > 3 && w < 11 && h < 11){

						    to_look_opp[look_opp_counter] = ((h + 2) * 15) + (w + 2);
						    look_opp_counter++;

						    to_look_opp[look_opp_counter] = ((h + 2) * 15) + (w);
						    look_opp_counter++;

						    to_look_opp[look_opp_counter] = ((h + 2) * 15) + (w - 2);
						    look_opp_counter++;

						    to_look_opp[look_opp_counter] = ((h ) * 15) + (w - 2);
						    look_opp_counter++;

						    to_look_opp[look_opp_counter] = ((h - 2) * 15) + (w - 2);
						    look_opp_counter++;

						    to_look_opp[look_opp_counter] = ((h - 2) * 15) + (w);
						    look_opp_counter++;

						    to_look_opp[look_opp_counter] = ((h - 2) * 15) + (w + 2);
						    look_opp_counter++;

                        }

					}
					
				}
			}
*/          
            to_look_mine[look_mine_counter] = (h * 15) + w;
			look_mine_counter++;

            to_look_opp[look_opp_counter] = (h * 15) + w;
			look_opp_counter++;

			
			//to_look_mine[look_mine_counter] = (h * 15) + w;
			//look_mine_counter++;
		}
	}
	//unsigned long long points[625];

	srand(time(NULL));
	for (int i = 0; i < 225; ++i)
	{
		srand(time(NULL));
		points[i] = rand() % 20;
	}

	get_points(data, to_look_mine, look_mine_counter, cur_player, points);
	get_points(data, to_look_opp, look_opp_counter, cur_player, points);

}


int get_best_move(double *data, int cur_player) {

	double points[625];

	for (int i = 0; i < 625; ++i)
	{
		points[i] = 0;
	}

	get_policy(data, cur_player, points);

	unsigned long long best = 0;
	int index = -5;
	for (int i = 0; i < 225; ++i)
	{	
		//printf("%lld\n", points[i]);
		if (data[transform_pos(i % 15, i / 15)] != 0){
			continue;
		}

		if (points[i] >= best){
			best = points[i];
			index = i;
		}
		if (points[i] >= 999999999)
		{
			if (data[transform_pos(i % 15, i / 15)] == 1){
				return -1;
			}
			if (data[transform_pos(i % 15, i / 15)] == 2){
				return -2;
			}
		}
	}


	return index;

}

int simulation(double *data, int cur_player) {
	int index = 0;

	while(index >= 0){
		index = get_best_move(data, cur_player);
		//printf("%d\n",index );
		if (index < 0){
			if (cur_player == 1){
				if (index == -1){
					return 1;
				} else {
					return 0;
				}
			}
			if (cur_player == 2){
				if (index == -2){
					return 1;
				} else {
					return 0;
				}
			}
		} else {
			data[transform_pos(index % 15, index / 15)] = cur_player;

			if (cur_player == 1){
				cur_player = 2;
			} else {
				cur_player = 1;
			}
		}
	}

	return index;
}


int  main(int argc, char const *argv[])
{
	double data[] = {
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		//--------------------------------------------------
		0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,      0,0,0,0,0,
		0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,      0,0,0,0,0,
		0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,      0,0,0,0,0,
		0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,      0,0,0,0,0,
		0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,      0,0,0,0,0,
		0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,      0,0,0,0,0,
		0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,      0,0,0,0,0,
		0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,      0,0,0,0,0,
		0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,      0,0,0,0,0,
		0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,      0,0,0,0,0,
		0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,      0,0,0,0,0,
		0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,      0,0,0,0,0,
		0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,      0,0,0,0,0,
		0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,      0,0,0,0,0,
		0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,      0,0,0,0,0,
		//------------------------------------------------
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	};

	int ans = simulation(data, 1);
	printf("%d\n", ans);

	//data[transform_pos(move % 15, move / 15)] = 2;

	for (int i = 0; i < 15; ++i)
	{
		for (int j = 0; j < 15; ++j)
		{
			printf("%.1lf ", data[transform_pos(j, i)]);
			
		}
		printf("\n");
	}
			printf("\n");



	return 0;
}
