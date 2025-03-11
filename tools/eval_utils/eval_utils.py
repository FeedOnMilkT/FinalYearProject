import pickle
import time

import numpy as np
import torch
import tqdm
import statistics

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


class InferenceTimeMeter:
    # Placeholder for inference time measurement
    def __init__(self):
        self.time_records = []
        self.pure_inf_time = []
        self.total_frames = 0
        self.start_time = 0
        self.warmup_frames = 10

    def start(self):
        torch.cuda.synchronize() # Synchronize CUDA
        self.start_time = time.time()

    def stop(self, is_warmup=False):
        torch.cuda.synchronize()
        elapsed = (time.time() - self.start_time) * 1000  # Convert to ms

        if not is_warmup:
            self.time_records.append(elapsed)
            self.total_frames += 1
        
        return elapsed

    def update_pure_inf_time(self, time_value):
        self.pure_inf_time.append(time_value)

    def get_statistics(self):
        if not self.time_records:
            return {}

        torch.cuda.synchronize()

        avg_time = statistics.mean(self.time_records)
        fps = 1000 / avg_time if avg_time > 0 else 0
        p95_latency = np.percentile(self.time_records, 95)

        pure_inf_avg = statistics.mean(self.pure_inf_time) if self.pure_inf_time else 0
        pure_inf_fps = 1000 / pure_inf_avg if pure_inf_avg > 0 else 0

        return {
            'average_time_ms': avg_time,
            'fps': fps,
            'p95_latency_ms': p95_latency,
            'pure_inference_time_ms': pure_inf_avg,
            'pure_inference_fps': pure_inf_fps,
            'total_frames_evaluated': self.total_frames
        }



def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    performance_mode =  getattr(args, 'infer_time', False)
    detailed_perf = getattr(args, 'detailed_perf', False)
    
    # In case of detailed performance measurement
    if performance_mode:
        inference_timer = InferenceTimeMeter()
        gpu_info = {}

        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            gpu_info = {
                'name: ': torch.cuda.get_device_name(current_device),
                'memory_allocated_start': torch.cuda.memory_allocated(current_device) / (1024**2),  # in MB
                'memory_cached_start': torch.cuda.memory_reserved(current_device) / (1024**2)  # in MB
            }

        logger.info('Running in performance evaluation mode. Warming up...')
    else:
        infer_time_meter = common_utils.AverageMeter()

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)

    # Evaluation Entire Time
    start_time = time.time()

    warmup_complete = False
    warmup_frames = 10 if performance_mode else 0

    for i, batch_dict in enumerate(dataloader):

        is_warmup = performance_mode and i < warmup_frames
        
        if performance_mode and i == warmup_frames:
            warmup_complete = True
            logger.info(f'Warmup complete after {warmup_frames} frames. Starting performance measurement...')
            inference_timer = InferenceTimeMeter()

        load_data_to_gpu(batch_dict)

        disp_dict = {}

        if performance_mode and (i >= warmup_frames or i == 0):
            inference_timer.start()

        with torch.no_grad():
            if performance_mode and (i >= warmup_frames or i == 0):
                torch.cuda.synchronize()
                t_start = time.time()

            pred_dicts, ret_dict = model(batch_dict)

            if performance_mode and (i >= warmup_frames or i == 0):
                torch.cuda.synchronize()
                pure_inf_time = (time.time() - t_start) * 1000  # Convert to ms
                inference_timer.update_pure_inf_time(pure_inf_time)

        if performance_mode and (i >= warmup_frames or i == 0):
            elapsed = inference_timer.stop(is_warmup=is_warmup)
            if not is_warmup:
                disp_dict['time_ms'] = f'{elapsed:.2f}'
                disp_dict['fps'] = f'{1000/elapsed:.2f}'
        elif getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        '''
        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)

        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'
        '''

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos

        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    '''
    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict
    '''

    if performance_mode and inference_timer.time_records:
        perf_stats = inference_timer.get_statistics()

        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            gpu_info['memory_allocated_end'] = torch.cuda.memory_allocated(current_device) / (1024**2)
            gpu_info['memory_cached_end'] = torch.cuda.memory_reserved(current_device) / (1024**2)
            gpu_info['memory_peak'] = torch.cuda.max_memory_allocated(current_device) / (1024**2) # in MB
            logger.info(f"GPU Info: {gpu_info}")

        logger.info('*************** Inference Performance Report *****************')
        logger.info(f"Average Processing Time: {perf_stats['average_time_ms']:.2f} ms")
        logger.info(f"Inference FPS: {perf_stats['fps']:.2f}")
        logger.info(f"95th Percentile Latency: {perf_stats['p95_latency_ms']:.2f} ms")
        logger.info(f"Pure Model Inference Time: {perf_stats['pure_inference_time_ms']:.2f} ms")
        logger.info(f"Pure Model Inference FPS: {perf_stats['pure_inference_fps']:.2f}")
        
        if gpu_info:
            logger.info(f"GPU: {gpu_info.get('name', 'Unknown')}")
            logger.info(f"GPU Memory Usage: {gpu_info.get('memory_allocated_end', 0):.2f} MB")
            logger.info(f"GPU Memory Peak: {gpu_info.get('memory_peak', 0):.2f} MB")
        
        logger.info(f"Batch Size: {cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU}")
        logger.info(f"Total Frames Evaluated: {perf_stats['total_frames_evaluated']}")
        logger.info('***************************************************************')
        
        # Add performance stats to return dict
        for k, v in perf_stats.items():
            ret_dict[f'performance/{k}'] = v
        
        # Add GPU info to return dict if available
        if gpu_info:
            for k, v in gpu_info.items():
                if isinstance(v, (int, float)):
                    ret_dict[f'performance/gpu_{k}'] = v
                            
    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict    

if __name__ == '__main__':
    pass
