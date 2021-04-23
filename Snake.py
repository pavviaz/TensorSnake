import math
import os
import sys
import random
import pygame
import SnakeIO
import numpy as np
import tensorflow as tf


class HaltCallback(tf.keras.callbacks.Callback):  # this can help to stop learning before epochs will end
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') <= 0.0001 and logs.get('accuracy') > 0.9999:
            print("\n\n\nReached 0.0001 loss value and 1 accuracy so cancelling training!\n\n\n")
            self.model.stop_training = True


def reshape(inp_arr):  # reshaping input data to specific shape
    return inp_arr.reshape(int(inp_arr.shape[0] / data_len), data_len, 1)


class Cube(object):  # the cube class, used to visualize snake and snack

    def __init__(self, start, dirnx=1, dirny=0, color=(255, 0, 0)):
        self.pos = start
        self.dirnx = 1
        self.dirny = 0
        self.color = color

    def move(self, dirnx, dirny):  # move cube with specific direction
        self.dirnx = dirnx
        self.dirny = dirny
        self.pos = (self.pos[0] + self.dirnx, self.pos[1] + self.dirny)

    def draw(self, surface, eyes=False):  # draw the cube on the screen
        dis = width // rows
        i = self.pos[0]
        j = self.pos[1]

        pygame.draw.rect(surface, self.color, (i * dis + 1, j * dis + 1, dis - 2, dis - 2))
        if eyes:  # draw eyes on the head block
            centre = dis // 2
            radius = 3
            circleMiddle = (i * dis + centre - radius, j * dis + 8)
            circleMiddle2 = (i * dis + dis - radius * 2, j * dis + 8)
            pygame.draw.circle(surface, (0, 0, 0), circleMiddle, radius)
            pygame.draw.circle(surface, (0, 0, 0), circleMiddle2, radius)


class Snake(object):  # snake class
    body = []
    turns = {}

    def __init__(self, color, pos):
        self.color = color
        self.head = Cube(pos, color=self.color)
        self.body.append(self.head)
        self.dirnx = -1  # -1 = Left ; 1 = Right ; 0 = vertical movement
        self.dirny = 0  # -1 = Up ; 1 = Down ; 0 = horizontal movement

    def move(self, move_handler):  # move snake
        global model, snack, data_len, glob_iter

        if move_handler:  # NN handles game
            inp_array = np.array([])
            log = ""

            for data in data_evaluate():  # get input data
                print(data, end=" ")
                log += str(data) + " "
                inp_array = np.append(inp_array, data)

            SnakeIO.add_to_log(log)  # send input to log
            prediction = model.predict(reshape(inp_array))  # NN predicts the direction of movement
            print("\n" + str(prediction))
            for pred in prediction:
                if (float(prediction.max())) == (float(prediction[0][0])):  # NN gave 0 --> Left
                    if self.dirnx != 1:
                        self.dirnx = -1
                        self.dirny = 0
                        self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
                elif (float(prediction.max())) == (float(prediction[0][1])):  # NN gave 1 --> Right
                    if self.dirnx != -1:
                        self.dirnx = 1
                        self.dirny = 0
                        self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
                elif (float(prediction.max())) == (float(prediction[0][2])):  # NN gave 2 --> Up
                    if self.dirny != 1:
                        self.dirnx = 0
                        self.dirny = -1
                        self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
                elif (float(prediction.max())) == (float(prediction[0][3])):  # NN gave 3 --> Down
                    if self.dirny != -1:
                        self.dirnx = 0
                        self.dirny = 1
                        self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        else:  # player handles game
            for event in pygame.event.get():
                # print(self.dirnx , self.dirny)
                if event.type == pygame.QUIT:
                    pygame.quit()
                keys = pygame.key.get_pressed()

                for key in keys:
                    if keys[pygame.K_LEFT] and self.dirnx != 1:
                        SnakeIO.add_to_log("\n0\n")
                        self.dirnx = -1
                        self.dirny = 0
                        self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
                    elif keys[pygame.K_RIGHT] and self.dirnx != -1:
                        SnakeIO.add_to_log("\n1\n")
                        self.dirnx = 1
                        self.dirny = 0
                        self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
                    elif keys[pygame.K_UP] and self.dirny != 1:
                        SnakeIO.add_to_log("\n2\n")
                        self.dirnx = 0
                        self.dirny = -1
                        self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
                    elif keys[pygame.K_DOWN] and self.dirny != -1:
                        SnakeIO.add_to_log("\n3\n")
                        self.dirnx = 0
                        self.dirny = 1
                        self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        #  Send NN prediction to log
        if self.dirnx == -1 and self.dirny == 0:
            print("LEFT")
            SnakeIO.add_to_log("\n0\n")
        if self.dirnx == 1 and self.dirny == 0:
            print("RIGHT")
            SnakeIO.add_to_log("\n1\n")
        if self.dirnx == 0 and self.dirny == -1:
            print("UP")
            SnakeIO.add_to_log("\n2\n")
        if self.dirnx == 0 and self.dirny == 1:
            print("DOWN")
            SnakeIO.add_to_log("\n3\n")

        #  turn snake
        for i, c in enumerate(self.body):
            pr = c.pos[:]
            if pr in self.turns:
                turn = self.turns[pr]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(pr)
            else:
                c.move(c.dirnx, c.dirny)

    def reset(self, pos):
        self.head = Cube(pos, color=self.color)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = -1
        self.dirny = 0

    def addCube(self):  # add cube to snake
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0] - 1, tail.pos[1]), color=self.color))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0] + 1, tail.pos[1]), color=self.color))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] - 1), color=self.color))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] + 1), color=self.color))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy

    def draw(self, surface):  # draw snake on screen
        for i, c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)


def draw_grid(w, rows, surface):  # draws a grid on the screen
    sizeBtwn = w // rows

    x = 0
    y = 0
    for l in range(rows):
        x = x + sizeBtwn
        y = y + sizeBtwn

        pygame.draw.line(surface, (255, 255, 255), (x, 0), (x, w))
        pygame.draw.line(surface, (255, 255, 255), (0, y), (w, y))


def redraw_window(surface, message):  # draw window and some info
    global s, snack
    surface.fill((0, 0, 0))
    s.draw(surface)
    snack.draw(surface)
    #  drawGrid(width, rows, surface)

    pygame.draw.line(surface, (255, 255, 255), (0, 500), (500, 500))
    font = pygame.font.Font("Recursive_Casual-Light.ttf", 12)
    text_surface = font.render(message, True, (255, 255, 255))
    surface.blit(text_surface, dest=(10, 512))

    pygame.display.update()


def random_snack(rows, item):  # generating snack in random position
    positions = item.body

    while True:
        x = random.randrange(rows)
        y = random.randrange(rows)
        if len(list(filter(lambda z: z.pos == (x, y), positions))) > 0:
            continue
        else:
            break

    return (x, y)


def learning():
    global s, model, i, inp, out, glob_iter, points_count, middle_points_temp, middle_points, game_count, \
        record, model_name

    buffer_inp, buffer_out = SnakeIO.inp_out()  # get array of "good" routes
    if len(buffer_inp) > 0:  # if they exist
        inp = np.append(inp, buffer_inp)  # save them
        out = np.append(out, buffer_out)
        if i >= learning_rate:
            middle_points = middle_points_temp // game_count  # recalculate middle points
            # reset
            game_count = middle_points_temp = 0
            glob_iter += 1
            i = 0

            # trainingStopCallback = haltCallback()
            print("TRAINING IN PROGRESS")
            model.fit(reshape(inp), out, epochs=500)  # training NN
            print("TRAINED")

            # add new {glob_iter}, {record}, {middle_points} to meta file
            data = SnakeIO.get_from_meta(False)
            SnakeIO.add_to_meta(
                data[0].replace("\n", str(glob_iter)) + " \n" + data[1].replace("\n", str(record)) + " \n" + data[
                    2].replace("\n", str(middle_points)) + " \n", mode="w")

            inp = np.array([])  # clear arrays
            out = np.array([])
    elif glob_iter == 0:
        model = SnakeIO.get_snake()  # this helps snake on first global iteration


def model_initialization(rewrite):
    global model, model_name, glob_iter, record, middle_points
    if rewrite:
        model.save_weights(model_name + "/")  # saving (rewriting) new weights
        SnakeIO.add_to_meta("\n\n\n", mode="w")  # clear the meta file
    else:
        if os.path.exists(model_name + "/"):
            glob_iter, record, middle_points = SnakeIO.get_from_meta(True)  # get data from meta file
            if glob_iter >= iterations_limit:
                raise AttributeError("global iterations >= iterations_limit. Check your config.txt")
            model.load_weights(model_name + "/")  # load weights
        else:
            raise IOError(f"no model {model_name} found!")


def restart_game():
    global snack, possible_steps, i, model_name, points_count, middle_points_temp, middle_points, game_count
    middle_points_temp += points_count
    points_count = 0
    game_count += 1  # +1 round in iteration
    SnakeIO.add_to_log("E")  # that is a sign the route lead snake to death (NN will not be trained this route)
    learning()  # save "good" routes and train NN if {i} >= {learning_rate}
    model.save_weights(model_name + "/")  # save trained model

    # reset
    SnakeIO.add_to_log("", mode="w")
    s.reset(snake_init_pos)
    snack = Cube(random_snack(rows, s), color=snack_color)
    possible_steps = possible_steps_const


def data_evaluate():
    global s, snack
    return [eval(expression) for expression in data_expr]


def main():
    global s, snack, model, i, possible_steps, glob_iter, model_name, \
        game_count, points_count, middle_points_temp, middle_points, record, rewrite_model, GPU_usage, snake_color, \
        snack_color, possible_steps_const, learning_rate, epochs, iterations_limit

    model_name, rewrite_model, GPU_usage, snake_color, snack_color, possible_steps_const, \
        learning_rate, epochs, iterations_limit = SnakeIO.config_parser()  # get constants from the config file
    SnakeIO.model_name = model_name

    # info
    print(
        f"!!! IF YOU GOT [op:RestoreV2] ERROR, JUST RESTART THE PROGRAM !!!\n"
        f"----config----\n"
        f"model_name = {model_name}\n"
        f"rewrite_model = {rewrite_model}\n"
        f"GPU_usage = {GPU_usage}\n"
        f"snake_color = {snake_color}\n"
        f"snack_color = {snack_color}\n"
        f"possible_steps_const = {possible_steps_const}\n"
        f"learning_rate = {learning_rate}\n"
        f"epochs = {epochs}\n"
        f"iterations_limit = {iterations_limit}\n"
        f"----config----\n"
        f"(c) pavviaz (https://github.com/pavviaz/TensorSnake), 2021\n"
        f"Press Enter to continue...")
    input()

    pygame.init()  # PyGame init
    clock = pygame.time.Clock()
    win = pygame.display.set_mode((width, width + info_width))  # window init

    SnakeIO.add_to_log("", mode="w")  # Clear logs
    SnakeIO.gpu(GPU_usage)  # True = using GPU, False = using CPU
    if GPU_usage:
        print("To run NN training on GPU, you should install CUDA and CuDNN drivers first!\nCheck this out: "
              "https://www.youtube.com/watch?v=hHWkvEcDBO0")

    model = SnakeIO.get_snake()  # get model
    model_initialization(rewrite_model)  # True = create(rewrite) model, False = load existing model

    s = Snake(snake_color, snake_init_pos)  # snake init (coords. 10;10)
    snack = Cube(random_snack(rows, s), color=snack_color)  # snack init (random coords)
    possible_steps = possible_steps_const  # snake can only make {possible_steps} steps to take snack, or it will die

    while glob_iter < iterations_limit:
        event = pygame.event.get()
        print(f"i = {i}\npos_steps = {possible_steps}\nglob_iter = {glob_iter}\nPress ESC to exit")

        pygame.time.delay(1)
        clock.tick(1000)

        s.move(True)  # True = AI handles game ; False = User handles game

        for e in event:  # catch ESC button to exit
            keys = pygame.key.get_pressed()
            for key in keys:
                if keys[pygame.K_ESCAPE]:
                    sys.exit()

        if len(s.body) - 1 > record:  # record update
            record = len(s.body) - 1

        if s.body[0].pos == snack.pos:  # snake ate snack
            points_count += 1  # +1 snack was eaten in round
            i += 1  # +1 snack was eaten in iteration
            SnakeIO.add_to_log("\n")  # that is a sign the "good" route (route lead snake to snack)
            s.addCube()  # snake gets longer
            snack = Cube(random_snack(rows, s), color=snack_color)  # respawn snack
            possible_steps = possible_steps_const  # restore {possible_steps}

        for x in range(len(s.body)):  # self-snake collision
            if s.body[x].pos in list(map(lambda z: z.pos, s.body[x + 1:])):
                restart_game()
                break

        if s.head.pos[0] in [x_min, x_max] or s.head.pos[1] in [y_min, y_max]:  # wall collision
            restart_game()

        if possible_steps == 0:  # too many steps
            restart_game()

        possible_steps -= 1
        redraw_window(win,
                      f"i = {i}, pos_steps = {possible_steps}, glob_iter = {glob_iter}, record = {record},"
                      f"g_c = {game_count}, p = {points_count},m_p = {middle_points}")


# constants

# these constants are distances from specific block to snack, tail or wall
block_west_snack = "round(math.sqrt((snack.pos[0] - s.head.pos[0] + 1) ** 2 + (snack.pos[1] - s.head.pos[1]) ** 2), 1)"
block_west_tail = "round(math.sqrt((s.body[-1].pos[0] - s.head.pos[0] + 1) ** 2 + (s.body[-1].pos[1] - s.head.pos[1]) " \
                  "** 2), 1) "
block_west_wall = "round(math.sqrt((x_min - s.head.pos[0] + 1) ** 2), 1)"
block_east_snack = "round(math.sqrt((snack.pos[0] - s.head.pos[0] - 1) ** 2 + (snack.pos[1] - s.head.pos[1]) ** 2), 1)"
block_east_tail = "round(math.sqrt((s.body[-1].pos[0] - s.head.pos[0] - 1) ** 2 + (s.body[-1].pos[1] - s.head.pos[1]) " \
                  "** 2), 1) "
block_east_wall = "round(math.sqrt((x_max - s.head.pos[0] - 1) ** 2), 1)"
block_north_snack = "round(math.sqrt((snack.pos[0] - s.head.pos[0]) ** 2 + (snack.pos[1] - s.head.pos[1] + 1) ** 2), 1)"
block_north_tail = "round(math.sqrt((s.body[-1].pos[0] - s.head.pos[0]) ** 2 + (s.body[-1].pos[1] - s.head.pos[1] + " \
                   "1) ** 2), 1) "
block_north_wall = "round(math.sqrt((y_min - s.head.pos[1] + 1) ** 2), 1)"
block_south_snack = "round(math.sqrt((snack.pos[0] - s.head.pos[0]) ** 2 + (snack.pos[1] - s.head.pos[1] - 1) ** 2), 1)"
block_south_tail = "round(math.sqrt((s.body[-1].pos[0] - s.head.pos[0]) ** 2 + (s.body[-1].pos[1] - s.head.pos[1] - " \
                   "1) ** 2), 1) "
block_south_wall = "round(math.sqrt((y_max - s.head.pos[1] - 1) ** 2), 1)"
block_northwest_snack = "round(math.sqrt((snack.pos[0] - s.head.pos[0] + 1) ** 2 + (snack.pos[1] - s.head.pos[1] + 1) " \
                        "** 2), 1) "
block_northwest_tail = "round(math.sqrt((s.body[-1].pos[0] - s.head.pos[0] + 1) ** 2 + (s.body[-1].pos[1] - " \
                       "s.head.pos[1] + 1) ** 2), 1) "
block_northwest_wall = "round(math.sqrt((x_min - s.head.pos[0] + 1) ** 2 + (y_min - s.head.pos[1] + 1) ** 2), 1)"
block_southwest_snack = "round(math.sqrt((snack.pos[0] - s.head.pos[0] + 1) ** 2 + (snack.pos[1] - s.head.pos[1] - 1) " \
                        "** 2), 1) "
block_southwest_tail = "round(math.sqrt((s.body[-1].pos[0] - s.head.pos[0] + 1) ** 2 + (s.body[-1].pos[1] - " \
                       "s.head.pos[1] - 1) ** 2), 1) "
block_southwest_wall = "round(math.sqrt((x_min - s.head.pos[0] + 1) ** 2 + (y_max - s.head.pos[1] - 1) ** 2), 1)"
block_northeast_snack = "round(math.sqrt((snack.pos[0] - s.head.pos[0] - 1) ** 2 + (snack.pos[1] - s.head.pos[1] + 1) " \
                        "** 2), 1) "
block_northeast_tail = "round(math.sqrt((s.body[-1].pos[0] - s.head.pos[0] - 1) ** 2 + (s.body[-1].pos[1] - " \
                       "s.head.pos[1] + 1) ** 2), 1) "
block_northeast_wall = "round(math.sqrt((x_max - s.head.pos[0] - 1) ** 2 + (y_min - s.head.pos[1] + 1) ** 2), 1)"
block_southeast_snack = "round(math.sqrt((snack.pos[0] - s.head.pos[0] - 1) ** 2 + (snack.pos[1] - s.head.pos[1] - 1) " \
                        "** 2), 1) "
block_southeast_tail = "round(math.sqrt((s.body[-1].pos[0] - s.head.pos[0] - 1) ** 2 + (s.body[-1].pos[1] - " \
                       "s.head.pos[1] - 1) ** 2), 1) "
block_southeast_wall = "round(math.sqrt((x_max - s.head.pos[0] - 1) ** 2 + (y_max - s.head.pos[1] - 1) ** 2), 1)"
data_expr = [block_west_snack, block_west_tail, block_west_wall,
             block_east_snack, block_east_tail, block_east_wall,
             block_north_snack, block_north_tail, block_north_wall,
             block_south_snack, block_south_tail, block_south_wall,
             block_northwest_snack, block_northwest_tail, block_northwest_wall,
             block_southwest_snack, block_southwest_tail, block_southwest_wall,
             block_northeast_snack, block_northeast_tail, block_northeast_wall,
             block_southeast_snack, block_southeast_tail, block_southeast_wall]

model_name = ""  # name of training model
rewrite_model = True  # True = create (or rewrite existing) model with {model_name} ; False = load model {model_name}
GPU_usage = False  # use GPU for training (only if specific drivers are installed)

width = 500  # screen width (and height)
info_width = 50  # additional space for some info on screen
rows = 20  # rows on game field
x_min = -1
x_max = rows
y_min = -1
y_max = rows
snake_color = (255, 0, 0)  # red
snack_color = (0, 255, 0)  # green
snake_init_pos = (10, 10)
possible_steps_const = 200  # max possible steps until snake death
learning_rate = 100  # if snake ate more food than {learning_rate} -->
# training starts with all saved "good" routes (route lead snake to snack)
iterations_limit = -1  # limit on a certain value of training iterations
# ( {learning_rate}*{iterations_limit} snacks will be eaten)
data_len = len(data_expr)  # length of input data (using for reshaping)
epochs = 500  # how many fitting iterations will NN go through

# variables
i = 0  # food eaten during one training iteration
glob_iter = 0  # count of all the training iterations
game_count = 0  # count of games during one training iteration
points_count = 0  # points per round
middle_points_temp = middle_points = 0  # middle point for all rounds
record = 0
inp = np.array([])
out = np.array([])

if __name__ == "__main__":
    main()
    print(f"Training ended after {glob_iter} iterations with {middle_points} middle_points and {record} record. "
          f"See {model_name}/meta.txt for more info")
