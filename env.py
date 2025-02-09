import numpy as np

RUNNING = 2
COMPLETED = 3
IDLE = 1


class Officers:
    def __init__(self, num_officier, num_event, num_task):
        # Total number of officer
        self.num_officer = num_officier
        self.num_event = num_event  # Total number of event
        self.num_task = num_task  # Total number of task per each event

        # initialize capability matrix for each event, each task
        self.capability = self.random_capability()
        # self.capability = example.officer_capability

    def random_capability(self):
        officers = []
        # Assign random capability for each officer
        for i in range(self.num_officer):
            capability = np.random.randint(1, 20, (self.num_event, self.num_task))
            officers.append(capability)
        return officers

    def reset_capability(self):
        self.capability = self.random_capability()


class Events:
    def __init__(self, num_event, num_task):
        self.num_event = num_event
        self.num_task = num_task
        self.num_problems = num_event * num_task
        # Initialize when each event start
        self.occurence = self.random_occurence()
        # self.occurence = example.occurence
        # Initialize transition matrix from base station to each event
        self.transition_matrix = self.random_transition()
        # self.transition_matrix = example.transition_matrix

    def random_occurence(self):
        occurence = np.random.randint(1, 20, self.num_event)
        occurence = np.sort(occurence)
        return occurence

    def random_transition(self):
        n = self.num_event + 1
        # Return a symetric matrix with 0 in the diagonal
        distrbution = np.random.randint(1, 20, int(n * (n - 1) / 2))
        matrix = np.zeros((n, n))
        matrix[np.triu_indices(n, 1)] = distrbution
        matrix[np.tril_indices(n, -1)] = matrix.T[np.tril_indices(n, -1)]
        return matrix

    def reset_events(self):
        self.occurence = self.random_occurence()
        self.transition_matrix = self.random_transition()


class EventsWithPriority(Events):
    def __init__(self, num_event, num_task, priority_type="score"):
        super().__init__(num_event, num_task)
        self.priority_type = priority_type
        # Introduce priority for each event
        if priority_type == "rank":
            # the higher the number, the higher the priority
            self.set_rank_priority(priority_list="random")
        elif priority_type == "score":
            # the highest score the
            self.set_score_priority()
        else:
            raise ("Unknown priority type")

    def set_rank_priority(self, priority_list):
        if priority_list == "random":
            self.priority = np.random.choice(range(1, self.num_event + 1), size=self.num_event, replace=False)
        else:
            if len(priority_list) != self.num_event:
                raise ("Priority list must have the same length with number of events")
            self.priority = priority_list

    def set_score_priority(self, priority_list):
        if priority_list == "random":
            self.priority = np.random.randint(0, 100, size=self.num_event)
        else:
            if len(priority_list) != self.num_event:
                raise ("Priority list must have the same length with number of events")
            self.priority = priority_list


class Environment:
    def __init__(self, officers, events):
        self.officers = officers
        self.events = events
        self.num_problems = events.num_problems
        # self.actual_capability = self.random_capability() # add noise to officers' capability
        self.actual_capability = self.officers.capability  # No noise
        self.reset(reset_officers=False, reset_events=False)
        self.obs = self.get_observation()
        self.event_time_end = []
        self.event_time_completed = []

    def random_capability(self):
        actual_capability = []
        ### Introduce gaussian randomness into the actual time taken
        for cap in self.officers.capability:
            mu, sigma = 0, 0.1
            add_ = np.random.normal(mu, sigma, cap.shape)
            actual_capability.append(add_ + cap)
        return actual_capability

    def reset_actual_capability(self):
        self.actual_capability = self.random_capability()

    def reset(self, reset_officers=True, reset_events=True):
        self.officer_occupied = [False] * self.officers.num_officer

        # codes Idle = 1, Running = 3, Over = 2
        # These are created for initialization and comparison when all tasks are done
        self.event_task_status_start = np.full((self.events.num_event, self.events.num_task), IDLE)
        self.event_task_status_end = np.full((self.events.num_event, self.events.num_task), COMPLETED)

        # To record the actual time of each task
        self.event_task_time_start = np.zeros((self.events.num_event, self.events.num_task))
        self.event_task_time_end = np.zeros((self.events.num_event, self.events.num_task))

        # In each event which task (task ID) is running. Position is event ID and value is Task ID
        self.running_task = [0] * self.events.num_event

        # reset officers
        if reset_officers:
            self.officers.reset_capability()

        # reset events
        if reset_events:
            self.events.reset_events()
            self.obs = self.get_observation()  # Update observation

        return self.get_observation()

    def step(self, assignment):
        #  Simulation
        # Input: assignment of the officers is a matrix
        #        occurence (time start of each event) from attribute
        #        actual capability from attribute
        #        transition matrix (time unit to move) from attribute

        # Step 1: Initialization
        # Each event will have a list/dictionary
        # [Task_number, start_time, end_time, officer_assigned]
        # [-1, -1, -1, -1] -> initialization value
        event_list = []
        for i in range(self.events.num_event):
            event_list.append({"task_id": -1, "start_time": -1, "end_time": -1, "officer_assigned": -1})

        # book-keeping which officer is working
        officer_occupied = [False] * self.officers.num_officer

        # book-keeping officer location: -1: Base-station, 0:event_0, ...
        officer_location = [-1] * self.officers.num_officer

        # keep track of status of event_task_status:
        event_task_status = self.event_task_status_start.copy()

        # get the sequence of event will happen
        occurence = {}
        for i, occur_time in enumerate(self.events.occurence):
            occurence[i] = occur_time

        # book-keeping for end-time of event
        end_time = {}
        for i in range(len(occurence)):
            end_time[i] = -1

        # Step 2: Simulation
        check_end = np.sum(self.event_task_status_end - event_task_status)
        time = 0
        tie = False

        # repeat until all tasks are done
        while check_end > 0:
            next_events_time = []
            next_events_indices = []
            for key, value in occurence.items():
                if value >= time:
                    next_events_time.append(value)
                    next_events_indices.append(key)

            for key, value in end_time.items():
                if value >= time:
                    next_events_time.append(value)
                    next_events_indices.append(key)

            assert (len(next_events_time) > 0)
            time = np.min(next_events_time)

            # Choose minimum from the filtered elements and extract Event ID.
            min_event_time = []
            min_event_indices = []
            for key, value in occurence.items():
                if value == time:
                    min_event_time.append(value)
                    min_event_indices.append(key)

            for key, value in end_time.items():
                if value == time:
                    min_event_time.append(value)
                    min_event_indices.append(key)

            # In the case that's there are more than 2 events with the same occuring time
            if len(min_event_time) > 1:
                tie = True
                min_event_index = self.get_event_tie(min_event_indices, event_list, event_task_status)
            else:
                tie = False
                min_event_index = min_event_indices[0]  # remember that events already sorted with time occurence

            # If Event(EventID) has not yet started, all tasks will have the status == 1 -> idle
            if event_task_status[min_event_index, 0] == IDLE:
                officier_assigned = assignment[min_event_index, 0]  # first task of the event
                task_id = event_list[min_event_index]["task_id"]
            # if Event(EventID) has started and is not in last task:
            elif event_task_status[min_event_index, -1] == IDLE:
                task_id = event_list[min_event_index]["task_id"]
                officier_assigned = assignment[min_event_index, task_id + 1]  # subsequent task
            else:
                officer_id = event_list[min_event_index]["officer_assigned"]
                officer_occupied[officer_id] = False
                del end_time[min_event_index]
                task_id = event_list[min_event_index]["task_id"]
                event_task_status[min_event_index, task_id] = COMPLETED
                check_end = np.sum(self.event_task_status_end - event_task_status)
                continue

            # Check officer occupied for Officer ID.
            if not officer_occupied[officier_assigned]:
                # If Event(EventID) has not yet started
                if event_task_status[min_event_index, 0] == IDLE:
                    # Remove occurence of Event(EventID) from List1
                    del occurence[min_event_index]
                    # Extract Officer ID location : Loc
                    location = officer_location[officier_assigned]
                    # Update the data
                    event_list[min_event_index]["task_id"] = task_id + 1
                    event_list[min_event_index]["start_time"] = time
                    working_time = self.actual_capability[officier_assigned][min_event_index, task_id + 1]
                    travelling_time = self.events.transition_matrix[location, min_event_index]
                    event_list[min_event_index]["end_time"] = time + working_time + travelling_time
                    event_list[min_event_index]["officer_assigned"] = officier_assigned
                    end_time[min_event_index] = event_list[min_event_index]["end_time"]
                    # Set Officer Occupied (Officer ID) = TRUE
                    officer_occupied[officier_assigned] = True
                    # Set Officer Location (Officer ID) = Event(EventID) Location
                    officer_location[officier_assigned] = min_event_index
                    # set event_task_status(EventId, Task number) = running
                    event_task_status[min_event_index, task_id + 1] = RUNNING
                # If event has been running
                else:
                    if event_list[min_event_index]["officer_assigned"] != -2:
                        previous_officer = event_list[min_event_index]["officer_assigned"]
                        # Set Officer Occupied (Previous_officer ID) = FALSE
                        officer_occupied[previous_officer] = False
                        # set event_task_status(EventID, Task number) = Completed
                        event_task_status[min_event_index, task_id] = COMPLETED

                    # Extract Officer ID location : Loc
                    location = officer_location[officier_assigned]
                    # Update the data
                    event_list[min_event_index]["task_id"] = task_id + 1
                    event_list[min_event_index]["start_time"] = time
                    working_time = self.actual_capability[officier_assigned][min_event_index, task_id + 1]
                    travelling_time = self.events.transition_matrix[location, min_event_index]
                    event_list[min_event_index]["end_time"] = time + working_time + travelling_time
                    event_list[min_event_index]["officer_assigned"] = officier_assigned
                    end_time[min_event_index] = event_list[min_event_index]["end_time"]
                    # Set Officer Occupied (Officer ID) = TRUE
                    officer_occupied[officier_assigned] = True
                    # Set Officer Location (Officer ID) = Event(EventID) Location
                    officer_location[officier_assigned] = min_event_index
                    # set event_task_status(EventId, Task number) = running
                    event_task_status[min_event_index, task_id + 1] = RUNNING
            # the officer is already occupied
            else:
                # Extract Officer ID assignment: Event(i)
                working_event = officer_location[officier_assigned]
                # If Event(EventID) has not yet started:
                if event_task_status[min_event_index, 0] == IDLE:
                    # set occurence(Event(Event ID)) = Event(i).EndTime
                    occurence[min_event_index] = event_list[working_event]["end_time"]
                # this event has been started
                else:
                    # Set Officer Occupied(Previous_officer ID) = FALSE
                    previous_officer = event_list[min_event_index]["officer_assigned"]
                    officer_occupied[previous_officer] = False
                    # set event_task_status(EventID, Task number) = Completed
                    task_id = event_list[min_event_index]["task_id"]
                    event_task_status[min_event_index, task_id] = COMPLETED
                    # set Event(EventID).End Time = Event(i).End Time
                    event_list[min_event_index]["end_time"] = event_list[working_event]["end_time"]
                    end_time[min_event_index] = event_list[min_event_index]["end_time"]
                    # set officer_assigned value in the dictionary to -1 (This is to indicate that this event is waiting)
                    event_list[min_event_index]["officer_assigned"] = -2

        event_time_end = []
        for i in range(len(event_list)):
            event_time_end.append(event_list[i]["end_time"])

        check_end = np.sum(self.event_task_status_end - event_task_status)
        assert check_end == 0  # make sure all events are completed

        event_time_completed = []
        for i in range(len(event_list)):
            event_time_completed.append(event_list[i]["end_time"] - event_list[i]["start_time"])

        self.event_time_end = event_time_end
        self.event_time_completed = event_time_completed

        return event_time_end

    def get_event_tie(self, min_event_indices, event_list, event_task_status):
        status = {}
        for index in min_event_indices:
            task_id = event_list[index]["task_id"]
            status[index] = event_task_status[index, task_id]

        # Exract EventID in priority of(Running > Idle > completed)
        for index in status:
            if status[index] == RUNNING:
                return index

        for index in status:
            if status[index] == IDLE:
                return index

        for index in status:
            if status[index] == COMPLETED:
                return index

        # safe-guard
        return None

    def get_observation(self) -> np.ndarray:
        officers_capability = np.concatenate([capability.flatten() for capability in self.officers.capability])
        occurence = self.events.occurence
        transition_matrix = self.events.transition_matrix.flatten()
        obs = np.concatenate([officers_capability, occurence, transition_matrix])
        return obs

    def get_length_obs(self):
        return len(self.obs)