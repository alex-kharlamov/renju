## Abstract
Cочетание [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) и [deep learning](https://en.wikipedia.org/wiki/Deep_learning) является "горячей" темой на сегодняшний день.
К примеру, существует статья  [Playing atari with deep reinforcement learning](http://arxiv.org/pdf/1312.5602v1.pdf). Также широко известна [AlphaGo](https://en.wikipedia.org/wiki/AlphaGo), программа, которая впервые победила человека, профессионально играющего в Go.
Подробнее об этом [здесь](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf). Данный проект заключался в изучении подходов, использующихся в AlphaGo, и их реализации при создание собственного алгоритма для игры в  [рендзю](https://en.wikipedia.org/wiki/Renju), правила игры в которую были нами модифицированы в процессе реализации своих агентов.

## Предыдущие работы
В качестве предыдущих работ, следует упомянуть только об агенте, разработанном в компании DeepMind - AlphaGo, который был пионером в области применения алгоритмов глубинных нейронных сетей, а также поиска в дерево Монте-Карло.
Учитывая его большие успехи, в ходе реализации проекта было решено использовать архитектуру, максимально близкую к [используемой](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf). Если описать ее кратко, то она заключается в:
* Обучению Supervised Learning Policy Network на ходах профессиональных игроков собранных из сети Интернет.
* Обучению Reinforcement Learning Policy Network, которая итерационно обучается на играх с различными версиями SL и RL policy, что позволяет значительно улучшить качество агента.
* Применения поиска в дереве Монте-Карло, с использованием быстрой Rollout Policy, которая применяется для быстрых симуляций игры.
* Обучения Value Network для оценки вероятности выигрыша игроков.


## Архитектура

### Базовая концепция
В ходе реализации всех вышеописанных моделей, было решено отказаться от Value Network, ввиду большой сложности обучения, а также малой полезности, что и было сделано в последних версиях агента. Также, в ходе сравнения различных модификаций Rollout Policy, было решено остановиться на эвристическом алгоритме, который выбирает лучшие ходы на основе базовых комбинаций, ввиду предположения о возможности дебютных сочетаний в игре Рендзю, что было подтверждено симуляциями с безоговорочной победой эвристической функции оценки позиций.
Остановимся поподробнее на реализациях конкретных сетей и их модификациях.


### Policy network
В качестве Policy Network было применено 2 сети с такой архитектурой:
![Model](https://image.ibb.co/c3h2dF/model.png)

Каждая из которых отвечала за крестики и нолики соответственно. Каждая из сетей была обучена на своей части данных, которые были получены из 13 различных источников в интернете, без учета возможной модификации правил в Рендзю.


### Reinforcement Learning
В качестве RL policy была применена таже архитектура, что и в SL Policy Network, за исключением последней активации, которая была заменена с SoftMax на линейную, ввиду особенностей применяемого [Double Dueling Q Network](https://arxiv.org/abs/1511.06581) метода. Для обучения был применен поэтапный алгоритм:
* Выбрать лучшего агента
* Провести серию игр со случайной предыдущей версией агента
* Дообучить агента на основе результатов игр
* Добавить новую версию агента в общий пул агентов

В качестве инициализации были применены веса SL Network.


### Monte Carlo Tree Search
Был применен базовый поиск в дереве Монте Карло с применением техники Upper Confidence Bound Applied to Trees(UCT).
Ввиду огромной вычислительной сложности, были проведены сравнения различных способов реализаций, с такими результатами:

Pure python
226 
326
319

Pure С
223 
346
293 


Pure_python numpy speedup
490
937
914

С numpy speedup
593
1001
956

Python jit mcts
455
777
791

Python jit all
409
653
681


Python numpy speedup improved randomize
3599
3969 
4120

C numpy speedup improved randomize
4111
3906
4381

*где цифры обозначают количество проведенных полноценных симуляций поиска в дереве за одну секунду для 3 запусков.

По результатам сравнения было решено остановиться на реализации поиска, написанного на языке С, который подключается через интерфейс Ctypes к основной логике, написанной на языке Python, что удобно реализуется компиляцией динамической библиотеки с необходимыми методами.



## Результаты
В ходе тестирования агента с другими реализациями, вышеописанный агент вошел в 5 лучших, тем не менее сохраняя достаточную конкурентноспособность для игры с человеком.
Ввиду примененной логики и отсутствию жесткого прессинга в атаке, агента целесообразно применять в игровых средах для интересной для пользователя игры с изменяемым уровнем сложности, что легко может быть подстроено путем регуляции времени, выделяемого на MCTS.

## Заключение

В ходе реализации агента было изучено и реализовано большое количество различных методов и идей, часть из которых, возможно, требует доработки, но общая концепция нейронных сетей и поиска в дереве Монте-Карло хорошо себя зарекомендовала ввиду гибкости и успешности применения для игр с большим пространством решений, таких как Го, Рендзю и т.д. Для дальнейшего развития необходимо более углубленно изучить возможности дообучения агента методами с подкреплением, что должно сгладить шероховатости качества примененной нами обучающей выборки и улучшить общую силу агента.


## Системные требования

Для обучения:
* 4 GPU Tesla k80(24 gb) ~ 2 месяца обучения
* 2TB RAM

Для игры:
* Процессор 2 Ghz
* Оперативная память - 6GB


## Инструкция по запуску
Для установки требуется:
1. Склонировать репозиторий.
2. Перейти в папку src ("cd src").
3. Скомпилировать библиотеку для Rollout(make).
4. Запустить агента(python3 ham_agent.py)
