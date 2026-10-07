import os
import json
import cv2
import numpy as np
from multi_cam_infrence import CameraWrapper, get_camera_info
import time




def resize_roi(roi, original_size, new_size):
   
    x, y, w, h = roi
    orig_w, orig_h = original_size
    new_w, new_h = new_size

    sx = new_w / orig_w
    sy = new_h / orig_h

    return (
        int(round(x * sx)),
        int(round(y * sy)),
        int(round(w * sx)),
        int(round(h * sy))
    )


def map_detected_to_master(master_points,
                           detected_points,
                           max_distance=30):
   

    if len(detected_points) == 0:
        return [(0, 0)] * len(master_points)

    detected = np.array(detected_points, dtype=np.float32)
    used = np.zeros(len(detected), dtype=bool)

    mapped = []

    for master in master_points:

        master = np.array(master, dtype=np.float32)

        best_idx = -1
        best_dist = np.inf

        for i, det in enumerate(detected):

            if used[i]:
                continue

            dist = np.linalg.norm(master - det)

            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx != -1 and best_dist <= max_distance:
            print(detected[best_idx].astype(int))
            mapped.append(detected[best_idx].astype(int).tolist())
            used[best_idx] = True
        else:
            mapped.append([0, 0])

    return mapped
class ModelManager:
    """Save/load a named model: master image, region ROI, template image
    and config (threshold, max_matches, scale_range, match centers...)."""

    def __init__(self, base_dir="models"):
        self.base_dir = base_dir

    def _paths(self, name):
        d = os.path.join(self.base_dir, name)
        return {
            "dir": d,
            "master": os.path.join(d, "master.png"),
            "config": os.path.join(d, "config.json"),
        }

    def exists(self, name):
        return os.path.isfile(self._paths(name)["config"])

    def save(self, name, frame, reg, temp_reg, template,
              threshold, scale_range, scale_steps, max_matches, matches=None):

        p = self._paths(name)
        os.makedirs(p["dir"], exist_ok=True)

        cv2.imwrite(p["master"], frame)

        templates = template if isinstance(template, list) else [template]
        temp_regs = temp_reg if isinstance(temp_reg, list) else [temp_reg]

        for i, tmpl in enumerate(templates):
            cv2.imwrite(os.path.join(p["dir"], f"template_{i}.png"), tmpl)

        centers = []
        if matches:
            for m in matches:
                x1, y1 = m["top_left"]
                x2, y2 = m["bottom_right"]
                centers.append({
                    "score": m["score"],
                    "scale": m["scale"],
                    "top_left": [x1, y1],
                    "bottom_right": [x2, y2],
                    "center": [(x1 + x2) // 2, (y1 + y2) // 2]
                })

        config = {
            "reg": list(reg),
            "temp_reg": [list(tr) for tr in temp_regs],
            "threshold": threshold,
            "scale_range": list(scale_range),
            "scale_steps": scale_steps,
            "max_matches": max_matches,
            "matches": centers
        }

        with open(p["config"], "w") as f:
            json.dump(config, f, indent=2)

        print(f"Model '{name}' saved to {p['dir']}")

    def load(self, name):
        p = self._paths(name)
        if not self.exists(name):
            return None

        with open(p["config"]) as f:
            config = json.load(f)

        frame = cv2.imread(p["master"])

        reg = tuple(config["reg"])
        temp_regs = [tuple(tr) for tr in config["temp_reg"]]
        region = frame[int(reg[1]):int(reg[1] + reg[3]), int(reg[0]):int(reg[0] + reg[2])]

        templates = []
        i = 0
        while True:
            tpath = os.path.join(p["dir"], f"template_{i}.png")
            if not os.path.isfile(tpath):
                break
            templates.append(cv2.imread(tpath))
            i += 1

        return {
            "frame": frame,
            "region": region,
            "templates": templates,
            "reg": reg,
            "temp_regs": temp_regs,
            "config": config
        }


class ParentModelManager:
    """A 'parent' model is just a named folder that groups together 1..N
    independent child models (each a full ModelManager entry with its own
    ROI, templates and config, selected freely on the full camera frame).

    Layout:
        models/
          <parent_name>/
            parent_config.json      -> {"children": ["child_1", "child_2", ...]}
            children/
              child_1/master.png, config.json, template_0.png ...
              child_2/...
    """

    def __init__(self, base_dir="models"):
        self.base_dir = base_dir

    def _parent_dir(self, parent_name):
        return os.path.join(self.base_dir, parent_name)

    def _parent_config_path(self, parent_name):
        return os.path.join(self._parent_dir(parent_name), "parent_config.json")

    def _children_dir(self, parent_name):
        return os.path.join(self._parent_dir(parent_name), "children")

    def exists(self, parent_name):
        return os.path.isfile(self._parent_config_path(parent_name))

    def get_child_manager(self, parent_name):
        """Returns a ModelManager scoped to this parent's children folder."""
        return ModelManager(base_dir=self._children_dir(parent_name))

    def load_parent_config(self, parent_name):
        if not self.exists(parent_name):
            return {"children": []}
        with open(self._parent_config_path(parent_name)) as f:
            return json.load(f)

    def save_parent_config(self, parent_name, children_list):
        os.makedirs(self._parent_dir(parent_name), exist_ok=True)
        with open(self._parent_config_path(parent_name), "w") as f:
            json.dump({"children": children_list}, f, indent=2)

    def add_child(self, parent_name, child_name):
        cfg = self.load_parent_config(parent_name)
        if child_name not in cfg["children"]:
            cfg["children"].append(child_name)
        self.save_parent_config(parent_name, cfg["children"])

    def list_children(self, parent_name):
        return self.load_parent_config(parent_name)["children"]

    def load_all_children(self, parent_name):
        """Loads every child model, in the order they were added."""
        cm = self.get_child_manager(parent_name)
        loaded = []
        for name in self.list_children(parent_name):
            data = cm.load(name)
            if data is not None:
                loaded.append((name, data))
            else:
                print(f"Warning: child '{name}' listed in parent but could not be loaded, skipping.")
        return loaded


def check_pin(matches=None,config=None,name=None):
    if matches and config is not None:
        master_coordinate=[]
        current_coordinate=[]
        master=config["config"]["matches"]
        w,h=config["reg"][2],config["reg"][3]
        mask=np.zeros((int(h*1),int(w*1),3),dtype=np.uint8)
        print("Master:",len(master))
        print("Current:",len(matches))
        # plt.figure(figsize=(8, 4))
        for i in range(len(master)):
            master_coordinate.append(master[i]["center"])
            if i<len(matches):
                m=matches[i]
                x1, y1 = m["top_left"]
                x2, y2 = m["bottom_right"]
                x,y = [(x1 + x2) // 2, (y1 + y2) // 2]
                current_coordinate.append([x,y])
        print("current_coordinate:",current_coordinate)
        current_coordinate=map_detected_to_master(master_coordinate,current_coordinate,max_distance=20)
        
        
        for i in range(len(master)):
            x,y= current_coordinate[i]
            if x==0 and y==0:
                color=(0,0,255)
                x,y=master[i]["center"]
            else:
                color=(255,0,0)
            
            # print(x,y)
            x_,y_=master[i]["center"]
            print("master center:",master[i]["center"],"current center:",[x,y])
            cv2.circle(mask,(x_+50,y_),5,(0,255,0),-1)
            cv2.circle(mask,(x+50,y),2,(100,100,255),-1)
            cv2.circle(mask,(x_+50,y_),15,color,2)


        # cv2.imshow(f"Check {'' if name is None else f'({name})'}", mask)
        

        
        #     plt.scatter(x, y, color='red', label='Points')

        # plt.plot(x, y, '-o', label='Line')

        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.grid(True)
        # plt.legend()
        # plt.show()
        return mask



def detect_multi_scale_templates(image,
                                 template,
                                 scale_range=(0.5, 1.5),
                                 scale_steps=20,
                                 threshold=0.8,
                                 max_matches=30,
                                 match_method=cv2.TM_CCOEFF_NORMED,
                                 nms_threshold=0.3,
                                 visualize=True,t=None,config=None,name=None):

    img_color = image.copy()
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    templates = template if isinstance(template, list) else [template]
    templates_gray = [cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) for t in templates]

    scales = np.linspace(scale_range[0], scale_range[1], scale_steps)

    candidates = []
    
    for tmpl_gray in templates_gray:

        t_h, t_w = tmpl_gray.shape

        for scale in scales:

            new_w = int(t_w * scale)
            new_h = int(t_h * scale)

            if new_w < 5 or new_h < 5:
                continue

            if new_w > img_gray.shape[1] or new_h > img_gray.shape[0]:
                continue

            resized_template = cv2.resize(
                tmpl_gray,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            )

            result = cv2.matchTemplate(
                img_gray,
                resized_template,
                match_method
            )

            result_copy = result.copy()

            while True:

                _, max_val, _, max_loc = cv2.minMaxLoc(result_copy)

                if max_val < threshold:
                    break

                x, y = max_loc

                candidates.append({
                    "score": float(max_val),
                    "scale": float(scale),
                    "top_left": (x, y),
                    "bottom_right": (x + new_w, y + new_h)
                })

                # Suppress neighborhood (same scale)
                x1 = max(0, x - new_w // 2)
                y1 = max(0, y - new_h // 2)
                x2 = min(result.shape[1], x + new_w // 2)
                y2 = min(result.shape[0], y + new_h // 2)

                result_copy[y1:y2, x1:x2] = -1

    # use the first template's height for row grouping tolerance 9315481653
    t_h = templates_gray[0].shape[0]

    if len(candidates) == 0:
        return [], img_color

    # ---------------- NMS -----------------

    boxes = []
    scores = []

    for m in candidates:
        x1, y1 = m["top_left"]
        x2, y2 = m["bottom_right"]

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(m["score"])

    idxs = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        score_threshold=threshold,
        nms_threshold=nms_threshold
    )

    if len(idxs) == 0:
        return [], img_color

    matches = [candidates[i] for i in idxs.flatten()]

    # Highest score first
    matches.sort(key=lambda x: x["score"], reverse=True)

    # Keep only top N
    matches = matches[:max_matches]

    # Sort for localization
    row_tol = t_h // 2

    matches.sort(
        key=lambda m: (
            round(m["top_left"][1] / row_tol),
            m["top_left"][0]
        )
    )

    output = img_color.copy()

    if visualize:

        for i, m in enumerate(matches):

            cv2.rectangle(
                output,
                m["top_left"],
                m["bottom_right"],
                (0, 255, 0),
                2
            )

            cv2.putText(
                output,
                str(i + 1),
                (m["top_left"][0], m["top_left"][1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
        if t is not None:
            print("Time taken:", time.time()-t)

        # cv2.imshow(f"Matches{'' if name is None else f' ({name})'}", output)

        mask=check_pin(matches,config,name=name)
        # output=cv2.resize(output,(800,200))
        # mask=cv2.resize(mask,(800,200))
        # combine_img=np.hstack([output,mask])


    return matches, output,mask




    #     _, max_val, _, max_loc = cv2.minMaxLoc(result)

    #     if max_val > best_score:
    #         best_score = max_val
    #         best_match = (max_loc, new_w, new_h, scale)

    # if best_match is None:
    #     print("No match found.")
    #     return None

    # (top_left, w, h, scale) = best_match
    # bottom_right = (top_left[0] + w, top_left[1] + h)

    # print(f"Best score: {best_score:.4f} | Scale: {scale:.2f} | Location: {top_left}")
    
        
    # if visualize:
    #     out = img_color.copy()
    #     if best_score < thresh/100:
    #         print("Match score below threshold.")
    #         cv2.rectangle(out, (0,0), (10,10), (0, 0, 255), 2)
    #     else:
    #         cv2.rectangle(out, top_left, bottom_right, (0, 255, 0), 2)

    #     cv2.imshow("Match", out)
    #     cv2.waitKey(0)

    # return {
    #     "score": best_score,
    #     "top_left": top_left,
    #     "bottom_right": bottom_right,
    #     "scale": scale
    # }


def _init_cameras():
    devs = get_camera_info(no=4)
    print(devs)

    cams = [CameraWrapper(dev) for dev in devs]

    cams[0].get_things(exposure=18000, work_mode="Exposure")
    # cams[0].get_things(gain=0, work_mode="Gain")
    # cams[0].get_things(work_mode='ColorGain',
    #                 rGain=200,
    #                 bGain=150,
    #                 gGain=100)

    # cams[0].get_things(work_mode="sharp")
    # cams[0].get_things(work_mode="contrast")
    return cams


def run_parent(parent_store, parent_name):
    """RUN MODE: parent already exists on disk. Load every child model and,
    on each 'd' key press, run detection for every child one by one."""

    children_data = parent_store.load_all_children(parent_name)
    if not children_data:
        print(f"Parent '{parent_name}' has no loadable children, aborting.")
        return

    print(f"Loaded parent '{parent_name}' with {len(children_data)} child model(s): "
          f"{[name for name, _ in children_data]}")
    print("Controls: 'd' = run all children one by one, 'q' = quit")

    cams = _init_cameras()

    while True:
        frame = cams[0].grab()
        raw = cv2.resize(frame, None, fx=0.6, fy=0.6)
        frame=raw.copy()
        for child_name, cd in children_data:
                reg = cd["reg"]
                
                x, y, w, h = [int(v) for v in reg]
                cv2.rectangle(raw, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        # cv2.imshow("Image", raw)
        k = cv2.waitKey(1)

        if k ==ord("c"):
        
                    cams[0].get_things(work_mode="auto_balance")
                    print("auto_balances")

        if k == ord("d"):
            print(f"\n=== Running parent '{parent_name}' "
                  f"({len(children_data)} child model(s)) ===")
            result_img=[]
            for child_name, cd in children_data:
                reg = cd["reg"]
                cfg = cd["config"]
                x, y, w, h = [int(v) for v in reg]

                frame_h, frame_w = frame.shape[:2]

                # Clamp the ROI to the current frame bounds instead of
                # crashing if the live frame size differs from when this
                # child was saved (different camera resolution/session).
                x0 = max(0, min(x, frame_w))
                y0 = max(0, min(y, frame_h))
                x1 = max(0, min(x + w, frame_w))
                y1 = max(0, min(y + h, frame_h))

                if x1 <= x0 or y1 <= y0:
                    print(f"\n--- Child model: '{child_name}' ---")
                    print(f"[{child_name}] SKIPPED: ROI {reg} is outside the current "
                          f"frame ({frame_w}x{frame_h}). This child was probably saved "
                          f"against a different camera resolution/orientation.")
                    continue

                if (x0, y0, x1 - x0, y1 - y0) != (x, y, w, h):
                    print(f"[{child_name}] Note: ROI {reg} was clamped to fit frame "
                          f"({frame_w}x{frame_h}) -> using ({x0},{y0},{x1-x0},{y1-y0})")

                region = frame[y0:y1, x0:x1]

                print(f"\n--- Child model: '{child_name}' ---")
                matches, output,mask = detect_multi_scale_templates(
                    image=region,
                    template=cd["templates"],
                    scale_range=tuple(cfg["scale_range"]),
                    scale_steps=cfg["scale_steps"],
                    threshold=cfg["threshold"],
                    max_matches=cfg["max_matches"],
                    visualize=True,
                    t=time.time(),
                    config=cd,name=child_name
                )
                result_img.append([output,[x, y, w, h],mask])
                print(f"[{child_name}] Detected: {len(matches)}")
            # combine=np.hstack([result_img[0][1],result_img[1][1]])
            # cv2.imshow("combine",combine)
            process_frame=np.ones_like(frame)
            if len(result_img)>0:
                for re in result_img:
                    frame[re[1][1]:re[1][1]+re[1][3],re[1][0]:re[1][0]+re[1][2]]=re[0]
                    process_frame[re[1][1]:re[1][1]+re[1][3],re[1][0]:re[1][0]+re[1][2]]=re[2]

                frame=cv2.resize(frame,None,fx=0.5,fy=0.5)
                process_frame=cv2.resize(process_frame,None,fx=0.5,fy=0.5)

                combine=np.vstack([frame,process_frame])
                # cv2.namedWindow("combine",cv2.WINDOW_NORMAL)
                # cv2.imshow("Image",combine)
                cv2.waitKey(0)
                # cv2.imshow("process_frame",process_frame)

        if k == ord("q"):
            break


def build_parent(parent_store, parent_name):
    """BUILD MODE: parent doesn't exist yet. Create 1..N children in the
    same live session: 'a' select ROI, 't' add template(s), 'd' test,
    's' save current child, 'n' start next child, 'f' finish parent."""

    print(f"No parent '{parent_name}' found. Building children interactively.")
    print("Controls: 'a'=select region  't'=add template  'd'=test detect  "
          "'s'=save child  'n'=start next child  'f'=finish parent  'q'=quit")

    child_manager = parent_store.get_child_manager(parent_name)

    reg = None
    temp_reg = None
    region = None
    template = None
    matches = None
    template_list = []
    temp_reg_list = []

    threshold = 0.80
    scale_range = (0.8, 1.5)
    scale_steps = 20
    max_matches = 20

    child_index = 1
    current_child_name = None
    current_child_saved = False

    cams = _init_cameras()

    while True:
        frame = cams[0].grab()
        frame = cv2.resize(frame, None, fx=0.6, fy=0.6)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray[gray < 250] = 0
        gray[gray > 0] = 255
        # cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
        # cv2.imshow("gray", gray)
        # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        k = cv2.waitKey(1)

        if k ==ord("c"):

            cams[0].get_things(work_mode="auto_balance")
            print("auto_balances")
        if k == ord("a"):
            cv2.namedWindow("Select_region", cv2.WINDOW_NORMAL)
            reg = cv2.selectROI("Select_region", frame)
            region = frame[int(reg[1]):int(reg[1] + reg[3]), int(reg[0]):int(reg[0] + reg[2])]
            cv2.destroyWindow("Select_region")

        if k == ord("t") and reg is not None:
            cv2.namedWindow("select_temp", cv2.WINDOW_NORMAL)
            temp_reg = cv2.selectROI("select_temp", region)
            template = region[int(temp_reg[1]):int(temp_reg[1] + temp_reg[3]),
                               int(temp_reg[0]):int(temp_reg[0] + temp_reg[2])]
            template_list.append(template)
            temp_reg_list.append(temp_reg)
            print(f"Template {len(template_list)} added to current child "
                  f"({current_child_name or f'child_{child_index}'})")
            cv2.destroyWindow("select_temp")

        display = frame.copy()
        if reg is not None:
            cv2.rectangle(display, (int(reg[0]), int(reg[1])),
                           (int(reg[0] + reg[2]), int(reg[1] + reg[3])), (0, 255, 0), 2)
        cv2.imshow("Image", display)

        if k == ord("d") and reg is not None and len(template_list) > 0:
            region = frame[int(reg[1]):int(reg[1] + reg[3]), int(reg[0]):int(reg[0] + reg[2])]

            matches, vis,mask = detect_multi_scale_templates(
                image=region,
                template=template_list,
                scale_range=scale_range,
                scale_steps=scale_steps,
                threshold=threshold,
                max_matches=max_matches,
                visualize=True, t=time.time(), config=None
            )
            print("Detected:", len(matches))
            cv2.waitKey(0)


        if k == ord("s") and reg is not None and len(template_list) > 0 and matches is not None:
            if current_child_name is None:
                default_name = f"child_{child_index}"
                name_in = input(f"Child name (default '{default_name}'): ").strip()
                current_child_name = name_in if name_in else default_name

            child_manager.save(current_child_name, frame, reg, temp_reg_list, template_list,
                                threshold, scale_range, scale_steps, max_matches, matches)
            parent_store.add_child(parent_name, current_child_name)
            current_child_saved = True
            print(f"Child '{current_child_name}' saved under parent '{parent_name}'")

        if k == ord("n"):
            if not current_child_saved:
                print("Current child isn't saved yet — press 's' first if you want to keep it.")
            else:
                print(f"Finished child '{current_child_name}'. Starting a new child...")
                child_index += 1
                current_child_name = None
                current_child_saved = False
                reg = None
                temp_reg = None
                region = None
                template = None
                matches = None
                template_list = []
                temp_reg_list = []

        if k == ord("f"):
            children_so_far = parent_store.list_children(parent_name)
            if len(children_so_far) == 0:
                print("No children saved yet — parent not finished. Press 's' to save one first.")
            else:
                print(f"Parent '{parent_name}' finished with children: {children_so_far}")
                break

        if k == ord("q"):
            break


if __name__ == "__main__":

    parent_store = ParentModelManager()
    parent_name = input("Parent model name: ").strip()

    if parent_store.exists(parent_name):
        run_parent(parent_store, parent_name)
    else:
        build_parent(parent_store, parent_name)