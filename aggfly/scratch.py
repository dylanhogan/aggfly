# def _sine_dd(frame, axis, ddargs):
#     # frame = frame.compute()
#     tmax = np.max(frame, axis=axis)
#     dd_shape = tmax.shape
#     tmax = tmax.flatten()
#     tmin = np.min(frame, axis=axis).flatten()
#     tavg = (tmax + tmin) / 2
#     alpha = (tmax - tmin) / 2
#     degree_day_list = []
#     if ddargs[2] == 0:
#         for threshold in ddargs[0:2]:
#             arr = np.full_like(tavg, fill_value=np.nan)
#             # print(arr.shape)
#             # print((threshold >= tmax).shape)
#             # print((threshold <= tmin).shape)
#             ii = (threshold >= tmax) #.compute_chunk_sizes()
#             arr[(ii,)] = np.zeros_like(arr)[ii].compute_chunk_sizes()
#             ii = (threshold <= tmin).compute_chunk_sizes()
#             arr[(ii,)] = (tavg - threshold)[ii].compute_chunk_sizes()
#             ii = ((threshold < tmax) * (threshold > tmin)).compute_chunk_sizes()
#             arr[(ii,)] = ((
#                 (tavg[ii] - threshold)
#                 * np.arccos(
#                     (2 * threshold - tmax[ii] - tmin[ii]) / (tmax[ii] - tmin[ii])
#                 )
#                 + (tmax[ii] - tmin[ii])
#                 * np.sin(
#                     np.arccos(
#                         (2 * threshold - tmax[ii] - tmin[ii]) / (tmax[ii] - tmin[ii])
#                     )
#                 )
#                 / 2
#             ) / np.pi).compute_chunk_sizes()
#             degree_day_list.append(arr)
#         degree_days = degree_day_list[0] - degree_day_list[1]
#     elif ddargs[2] == 1:
#         for threshold in ddargs[0:2]:
#             arr = np.full_like(tavg, fill_value=np.nan)
#             arr[(threshold >= tmax)] = (threshold - tavg)[(threshold >= tmax)]
#             arr[(threshold <= tmin)] = np.zeros_like(arr)[(threshold <= tmin)]
#             ii = (threshold < tmax) & (threshold > tmin)
#             arr[ii] = (1 / (np.pi)) * (
#                 (threshold - tavg[ii])
#                 * (
#                     np.arctan(
#                         ((threshold - tavg[ii]) / alpha[ii])
#                         / np.sqrt(1 - ((threshold - tavg[ii]) / alpha[ii]) ** 2)
#                     )
#                     + (np.pi / 2)
#                 )
#                 + alpha[ii]
#                 * np.cos(
#                     (
#                         np.arctan(
#                             ((threshold - tavg[ii]) / alpha[ii])
#                             / np.sqrt(1 - ((threshold - tavg[ii]) / alpha[ii]) ** 2)
#                         )
#                     )
#                 )
#             )
#             degree_day_list.append(arr)
#         degree_days = degree_day_list[1] - degree_day_list[0]
#     else:
#         raise ValueError("Invalid degree day type")

#     return degree_days.reshape(dd_shape)

#     dask                      2023.12.1


# # def _sine_dd(frame, axis, ddargs):
# #     tmax = np.max(frame, axis=axis)
# #     d_shape = tmax.shape
# #     tmax = tmax.flatten()
# #     tmin = np.min(frame, axis=axis).flatten()
# #     tavg = (tmax + tmin) / 2
# #     alpha = (tmax - tmin) / 2
# #     output = np.full(tmax.shape, np.nan)
# #     threshold = ddargs[0]
# #     for i in np.arange(tmax.shape[0]):
# #         if (threshold >= tmax[i]):
# #             output[i] = 0
# #         elif (threshold <= tmin[i]):
# #             output[i] = (tavg[i] - threshold)
# #         else:
# #             output[i] = (
# #                 (tavg[i] - threshold)
# #                 * np.arccos(
# #                     (2 * threshold - tmax[i] - tmin[i]) / (tmax[i] - tmin[i])
# #                 )
# #                 + (tmax[i] - tmin[i])
# #                 * np.sin(
# #                     np.arccos(
# #                         (2 * threshold - tmax[i] - tmin[i]) / (tmax[i] - tmin[i])
# #                     )
# #                 )
# #                 / 2
# #             ) / np.pi
# #     return output.reshape(d_shape)